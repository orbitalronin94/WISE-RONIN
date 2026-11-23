"""Aggregate the 18 paired 100-step candidate artifacts without dropping failures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

BASE_PROFILE = {
    "threads": 4, "interop_threads": 2, "mkldnn_enabled": True,
    "foreach": False, "cpu_affinity": True, "delta_pre_normalize": False,
    "homeostasis_fused_scatter": False, "delta_parallel_refinements": 2,
    "organs": {
        "pain": {"response": 0.30, "decay": 0.55},
        "pleasure": {"response": 0.75},
        "pain_budget_gain": 0.10, "pleasure_budget_gain": 0.1125,
    },
}

def merge_delta(target, delta, path=""):
    conflicts = []
    for key, value in delta.items():
        here = f"{path}.{key}" if path else key
        if key not in target:
            target[key] = value
        elif isinstance(target[key], dict) and isinstance(value, dict):
            conflicts.extend(merge_delta(target[key], value, here))
        elif target[key] != value:
            conflicts.append({"key": here, "existing": target[key], "candidate": value})
    return conflicts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.root.glob("**/comparison.json"))
    reports = [
        json.loads(p.read_text(encoding="utf-8"))
        | {"artifact_path": str(p.parent.relative_to(args.root))}
        for p in files
    ]
    approved = [r for r in reports if r.get("decision") == "APPROVED"]
    pending = [r for r in reports if r.get("decision") != "APPROVED"]

    combined = json.loads(json.dumps(BASE_PROFILE))
    conflicts = []
    for report in approved:
        found = merge_delta(combined, report.get("profile_delta", {}))
        conflicts.extend([{"candidate": report["candidate"], **x} for x in found])
    if conflicts:
        combined = None

    approved_payload = {
        "approved_candidates": [r["candidate"] for r in approved],
        "pending_candidates": [
            {
                "candidate": r.get("candidate"),
                "reason": r.get("reasons") or [r.get("reason", "sin motivo registrado")],
                "artifact_path": r.get("artifact_path"),
            }
            for r in pending
        ],
        "combination_conflicts": conflicts,
        "combined_profile": combined,
        "training_authorized": bool(approved) and not conflicts,
        "training_note": (
            "La configuración combina únicamente los candidatos aprobados. "
            "La base de señales compartidas es común a todos los candidatos de esta rama."
        ),
    }
    (args.root / "approved_candidates.json").write_text(
        json.dumps(approved_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Resultado — torneo neuroplástico de 18 cambios",
        "",
        f"- Evaluaciones recibidas: **{len(reports)}/18**",
        f"- Aprobados: **{len(approved)}**",
        f"- Pendientes: **{len(pending)}**",
        f"- Perfil combinado sin conflictos: **{'sí' if combined is not None else 'no'}**",
        "",
        "Cada artefacto conserva main-control.log, candidate.log, los perfiles JSON y comparison.json.",
        "La aprobación exige estrictamente menor loss media, más pasos/segundo y dead no superior al control.",
        "Los pendientes se conservan; no se interpretan como descartados.",
        "",
        "## Aprobados",
        "",
    ]
    if approved:
        for r in approved:
            lines.append(
                f"- **{r['candidate']}** — loss {r['candidate_result']['mean_loss_100_steps']:.8f} "
                f"vs {r['control']['mean_loss_100_steps']:.8f}; "
                f"{r['candidate_result']['steps_per_second']:.6f} vs "
                f"{r['control']['steps_per_second']:.6f} pasos/s; "
                f"dead máx. {r['candidate_result']['max_dead_percent']:.4f}%."
            )
    else:
        lines.append("- Ninguno por ahora.")
    lines += ["", "## Pendientes", ""]
    if pending:
        for r in pending:
            reasons = r.get("reasons") or [r.get("reason", "sin motivo registrado")]
            lines.append(
                f"- **{r.get('candidate', 'desconocido')}** — {'; '.join(reasons)}. "
                f"Logs: {r.get('artifact_path')}/."
            )
    else:
        lines.append("- Ninguno.")
    if conflicts:
        lines += ["", "## Conflictos al combinar aprobados", ""]
        for c in conflicts:
            lines.append(
                f"- {c['candidate']}: {c['key']} ya vale {c['existing']}, "
                f"candidato propone {c['candidate']}."
            )
    lines += [
        "",
        "## Nota metodológica",
        "",
        "El control ejecuta main; cada candidato ejecuta el HEAD de esta PR con la base de señales compartidas "
        "y un único delta funcional. Por tanto, un aprobado acredita ese paquete (base compartida + delta), "
        "no una atribución causal aislada del delta respecto a la base compartida.",
        "",
        "ELM queda fuera de este torneo: es un probe analítico offline, no modifica el camino de entrenamiento "
        "autoregresivo y no puede mejorar la loss del trainer en esta comparación. Requiere un benchmark auxiliar.",
        "",
    ]
    (args.root / "AGGREGATE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
