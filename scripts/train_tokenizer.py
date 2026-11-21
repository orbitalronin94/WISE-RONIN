#!/usr/bin/env python3
"""Train the frozen 4k ByteLevel BPE tokenizer from the final train JSONL only."""
import argparse, json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
SPECIAL=["[PAD]","[UNK]","[BOS]","[EOS]","[SEP]"]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",default="data/train_25k.jsonl"); p.add_argument("--output",default="data/tokenizer/tokenizer.json"); p.add_argument("--vocab-size",type=int,default=4096); a=p.parse_args()
 texts=[]
 with Path(a.data).open(encoding="utf-8") as f:
  for line in f:
   if line.strip():
    r=json.loads(line); texts.append((str(r["prompt"])+"\n"+str(r["completion"])).strip())
 if not texts: raise ValueError("No training text")
 tok=Tokenizer(models.BPE(unk_token="[UNK]")); tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder=decoders.ByteLevel()
 trainer=trainers.BpeTrainer(vocab_size=a.vocab_size,min_frequency=2,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
 tok.train_from_iterator(texts,trainer=trainer)
 tok.post_processor=processors.TemplateProcessing(single="[BOS] $A [EOS]",pair="[BOS] $A [SEP] $B:1 [EOS]",special_tokens=[("[BOS]",tok.token_to_id("[BOS]")),("[EOS]",tok.token_to_id("[EOS]")),("[SEP]",tok.token_to_id("[SEP]"))])
 if tok.token_to_id("[PAD]")!=0: raise RuntimeError("PAD must be ID 0")
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); tok.save(str(out))
 print(json.dumps({"vocab_size":tok.get_vocab_size(),"pad_id":tok.token_to_id("[PAD]"),"path":str(out)}))
if __name__=="__main__": main()
