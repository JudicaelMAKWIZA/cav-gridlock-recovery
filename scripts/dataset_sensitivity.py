"""Calibration-only sensitivity audit; never changes the frozen production contract."""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from cav_recovery.dataset.contract import Contract
from cav_recovery.dataset.pipeline import read_rows, records, write_json
from cav_recovery.dataset.qualify import qualify


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    campaign=json.loads((args.calibration/"campaign.json").read_text())
    if campaign["phase"]!="calibration":
        parser.error("Sensitivity must use calibration episodes, never production or final test.")
    base=Contract(**campaign["contract"])
    cache=[]
    for folder,row in records(args.calibration):
        cache.append((row,json.loads((folder/"metadata.json").read_text()),list(read_rows(folder/"reference.jsonl.gz"))))
    results=[]
    for persistence in (3.,5.,8.):
        for progress in (.5,1.,1.5):
            for followup in (15.,30.,35.):
                contract=replace(base,persistence_s=persistence,progress_vehicle_lengths=progress,nominal_followup_s=followup)
                counts=Counter();families={}
                for row,metadata,frames in cache:
                    result=qualify(frames,metadata,row["integrity"],contract)
                    counts[result["label"]]+=1
                    families.setdefault(row["family"],Counter())[result["label"]]+=1
                results.append(dict(contract=contract.record(),labels=dict(counts),families=families))
    output=args.output or args.calibration/"sensitivity_extended.json"
    audit=dict(calibration_only=True,production_contract_unchanged=True,base_contract_sha256=base.sha256,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),episodes=len(cache),results=results)
    write_json(output,audit)
    print(json.dumps(dict(episodes=len(cache),parameter_combinations=len(results),output=str(output))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
