# WISE-RONIN — Manual QA review

The uploaded file contains 976 records. This review does not rewrite the source. It quarantines records that should not enter SFT until their wording or answer is corrected. The merge script keys most cases by `source_record` so differently phrased duplicates of the same claim can be reviewed independently without discarding unrelated examples.

## Review classes

| Class | Records | Finding / required action |
|---|---:|---|
| Electrical conductivity | 447 | Silver is the best electrical conductor among metals; copper is commonly second. The supplied answer reverses the ranking. |
| Capitals and political status | 326, 334, 345, 371 | Jerusalem needs geopolitical context; Almaty is no longer Kazakhstan's capital (Astana); Sri Lanka's legislative capital is Sri Jayawardenepura Kotte while Colombo is the commercial centre and seat of the president and government; Bolivia distinguishes constitutional capital Sucre from the seat of government La Paz. |
| Longest river | 38, 423, 644, 833, 960 | Nile is often listed first, but the Nile/Amazon comparison depends on source and mouth definitions and measurement method. Avoid presenting the superlative as uncontested. |
| Air composition | 39, 99, 424, 645, 834, 961 | The question asks what gas we breathe; the answer should be air, a mixture containing about 21% oxygen, not pure oxygen. |
| Evolution | 476, 659, 866 | Darwin and Alfred Russel Wallace independently developed the theory of evolution by natural selection. Avoid sole attribution. |
| Relativity | 477, 660, 867 | Einstein formulated the theory of relativity; “wrote the relativity” is imprecise. |
| Printing | 485, 663, 869 | Printing and movable type predate Gutenberg in East Asia. Specify Gutenberg's development of European movable-metal-type printing. |
| Radio history | 486 | “First radio signal” is underspecified and depends on what counts as a signal/transmission and which earlier experiments are included. |
| Microscopy | 490 | Leeuwenhoek was not the first microscope user; he is known for high-quality single-lens microscopes and important observations. |
| Thermometry | 492 | Galileo developed a thermoscope; the history of the thermometer involved several contributors, including Santorio. Do not state simply that Galileo invented the thermometer. |
| Cuban abolition | 507 | The prompt asks about Cuba but the completion says Spain. The answer must match the country asked about (abolition in Cuba: 1886). |
| Penicillin attribution | 868 | The answer is too elliptical for a factual QA item (“Fleming la vio”). Specify the 1928 discovery and name Alexander Fleming clearly. |
| Constitutional reference | 670, 873 | The prompt says only “the Constitution” while the completion assumes Spain. Name the Spanish Constitution in the question. |

## Source notes

- Kazakhstan's Constitution, Article 2: the capital is Astana — https://aifc.kz/wp-content/uploads/2024/06/constitution-of-the-rk-eng-1.pdf
- Spain's Ministry of Foreign Affairs, Sri Lanka country sheet (July 2025): Sri Jayawardenepura Kotte is legislative capital; Colombo is commercial capital and seat of the presidency and government — https://www.exteriores.gob.es/Documents/FichasPais/SRILANKA_FICHA%20PAIS.pdf
- United Nations document on Bolivia: Sucre is constitutional capital; La Paz is the seat of government — https://documents.un.org/access.nsf/get?DS=CMW%2FC%2FBOL%2F1&Lang=E&Open=
- Linnean Society history in *Biological Journal of the Linnean Society*: Darwin and Wallace's independent work — https://academic.oup.com/biolinnean/article/137/3/556/6693379
- Cambridge University Press, history of printing: East Asian origins and Gutenberg's European development — https://www.cambridge.org/core/books/abs/five-innovations-that-changed-human-history/inventing-printing/9A13F759425814CA7D3FC869B146BE04
- Whipple Museum, University of Cambridge, early thermometers: Galileo's thermoscope and the later thermometer development — https://www.whipplemuseum.cam.ac.uk/explore-whipple-collections/meteorology/early-thermometers-and-temperature-scales
- Rice University Galileo Project, thermometer history — https://galileo.library.rice.edu/sci/instruments/thermometer.html
- IUPAC Gold Book, dry-air composition: nitrogen 78.084%, oxygen 20.946% — https://goldbook.iupac.org/terms/view/C01214/html
- Guinness World Records, longest river: Nile is credited, while the Nile/Amazon comparison depends on definitions — https://www.guinnessworldrecords.com/world-records/longest-river
- Research paper on river-length measurement uncertainty — https://doi.org/10.1080/17538940902746082

## Policy

- Do not silently rewrite the uploaded source.
- Quarantine only identified records; preserve their original prompt, completion, source name, and source record ID in the audit JSONL.
- Correct or replace a quarantined item only after review, then regenerate the master corpus and its audit artifacts.
- The 32 quarantined records are excluded from training; the remaining accepted records retain source provenance.
