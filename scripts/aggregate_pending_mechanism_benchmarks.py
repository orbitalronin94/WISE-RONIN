"""Aggregate the 20 paired 100-step pending-mechanism artifacts without dropping failures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

EXPECTED_CANDIDATES = [
    "01_tagging_capture",
    "02_intrinsic_threshold",
    "03_simulated_pruning",
    "04_tag_protected_ltd",
    "05_one_shot_memory",
    "06_homeostatic_scaling",
    "07_prioritized_replay",
    "08_sleep_downscaling",
    "09_tag_plus_intrinsic",
    "10_tag_plus_pruning",
    "11_tag_plus_one_shot",
    "12_tag_plus_scaling",
    "13_tag_plus_replay",
    "14_tag_plus_downscale",
    "15_intrinsic_plus_pruning",
    "16_one_shot_plus_scaling",
    "17_pruning_plus_ltd",
    "18_scaling_plus_downscale",
    "19_tag_replay_downscale",
    "20_all_pending"
]


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
    names = [r.get("candidate") for r in reports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    unknown = sorted(set(names) - set(EXPECTED_CANDIDATES))
    if duplicates:
        raise ValueError(f"duplicate candidate reports: {duplicates}")
    if unknown:
        raise ValueError(f"unexpected candidate reports: {unknown}")
    by_name = {r.get("candidate"): r for r in reports}
    missing = [name for name in EXPECTED_CANDIDATES if name not in by_name]
    pending_missing = [
        {
            "candidate": name,
            "decision": "PENDING",
            "reasons": ["no llegó artefacto; evaluación fallida o no ejecutada"],
            "artifact_path": None,
        }
        for name in missing
    ]
    raw_approved = [r for r in reports if r.get("decision") == "APPROVED"]
    # A combination cannot promote a mechanism whose isolated hypothesis failed.
    requirements = {
        "04_tag_protected_ltd": ["01_tagging_capture"],
        "09_tag_plus_intrinsic": ["01_tagging_capture", "02_intrinsic_threshold"],
        "10_tag_plus_pruning": ["01_tagging_capture", "03_simulated_pruning"],
        "11_tag_plus_one_shot": ["01_tagging_capture", "05_one_shot_memory"],
        "12_tag_plus_scaling": ["01_tagging_capture", "06_homeostatic_scaling"],
        "13_tag_plus_replay": ["01_tagging_capture", "07_prioritized_replay"],
        "14_tag_plus_downscale": ["01_tagging_capture", "08_sleep_downscaling"],
        "15_intrinsic_plus_pruning": ["02_intrinsic_threshold", "03_simulated_pruning"],
        "16_one_shot_plus_scaling": ["05_one_shot_memory", "06_homeostatic_scaling"],
        "17_pruning_plus_ltd": ["03_simulated_pruning", "01_tagging_capture", "04_tag_protected_ltd"],
        "18_scaling_plus_downscale": ["06_homeostatic_scaling", "08_sleep_downscaling"],
        "19_tag_replay_downscale": ["01_tagging_capture", "07_prioritized_replay", "08_sleep_downscaling"],
        "20_all_pending": EXPECTED_CANDIDATES[:8],
    }
    raw_approved_names = {r.get("candidate") for r in raw_approved}
    approved, combination_blocked = [], []
    for report in raw_approved:
        required = requirements.get(report.get("candidate"), [])
        missing_components = [name for name in required if name not in raw_approved_names]
        if missing_components:
            report = dict(report)
            report["decision"] = "PENDING"
            report["reasons"] = [f"la combinación no puede promoverse: componentes individuales no aprobados ({', '.join(missing_components)})"]
            combination_blocked.append(report)
        else:
            approved.append(report)
    pending = [r for r in reports if r.get("decision") != "APPROVED"] + combination_blocked + pending_missing

    # Start from main's committed runtime profile, not a hand-maintained approximation.
    main_profile_path = args.root / "main-runtime-profile.json"
    if not main_profile_path.exists():
        raise FileNotFoundError(
            "main-runtime-profile.json must be copied from main/config/lumen_cpu_profile.json"
        )
    combined = json.loads(main_profile_path.read_text(encoding="utf-8"))
    conflicts = []
    for report in approved:
        found = merge_delta(combined, report.get("profile_delta", {}))
        conflicts.extend([{"candidate": report["candidate"], **x} for x in found])
    if conflicts:
        combined = None

    approved_payload = {
        "expected_candidates": EXPECTED_CANDIDATES,
        "received_candidates": sorted(by_name),
        "missing_candidates": missing,
        "approved_candidates": [r["candidate"] for r in approved],
        "raw_screening_passes": sorted(raw_approved_names),
        "combination_blocked": [r["candidate"] for r in combination_blocked],
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
        # A 100-step screen is never sufficient authorization for a training run.
        # Every apparent winner must first survive longer paired replication and holdout.
        "training_authorized": False,
        "training_note": (
            "Este torneo es solo cribado. Ningún resultado autoriza entrenamiento ni merge: "
            "los candidatos que pasen requieren réplica emparejada más larga, revisión de "
            "interacciones y validación de estabilidad/holdout."
        ),
    }
    (args.root / "approved_candidates.json").write_text(
        json.dumps(approved_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Resultado — torneo de 20 hipótesis sobre mecanismos pendientes",
        "",
        f"- Artefactos recibidos: **{len(reports)}/20**",
        f"- Evaluaciones contabilizadas: **{len(reports) + len(missing)}/20**",
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
            artifact_path = r.get("artifact_path")
            log_note = f"Logs: {artifact_path}/." if artifact_path else "Sin artefacto; revisar ejecución."
            lines.append(
                f"- **{r.get('candidate', 'desconocido')}** — {'; '.join(reasons)}. "
                f"{log_note}"
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
        "El control ejecuta main; cada candidata ejecuta el mismo código de entrenamiento con el perfil congelado "
        "de main más un único delta funcional. La comparación estima el efecto de ese delta bajo este "
        "checkpoint y este cribado; no demuestra causalidad general ni estabilidad a largo plazo.",
        "",
        "ELM queda fuera de este torneo: es un probe analítico offline, no modifica el camino de entrenamiento "
        "autoregresivo y no puede mejorar la loss del trainer en esta comparación. Requiere un benchmark auxiliar.",
        "",
    ]
    (args.root / "AGGREGATE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
