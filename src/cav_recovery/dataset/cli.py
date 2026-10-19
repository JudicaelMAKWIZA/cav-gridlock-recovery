"""Dataset CLI, separate from the simulation demos and SUMO commands."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(prog="cgr dataset",description="Generate and independently qualify physical CAV episodes.")
    commands=parser.add_subparsers(dest="command",required=True)
    sources=commands.add_parser("sources");sources.add_argument("--probe-native",action="store_true")
    generate=commands.add_parser("generate")
    generate.add_argument("--output-dir",type=Path,required=True)
    generate.add_argument("--phase",choices=("production","calibration"),default="production")
    generate.add_argument("--contract",type=Path)
    generate.add_argument("--count",type=int,default=0)
    generate.add_argument("--positive",type=int,default=0)
    generate.add_argument("--negative",type=int,default=0)
    generate.add_argument("--max-candidates",type=int,default=4000)
    generate.add_argument("--workers",type=int,choices=(1,2,3,4),default=2)
    generate.add_argument("--max-seconds",type=float,default=7200)
    generate.add_argument("--max-gib",type=float,default=4.)
    generate.add_argument("--reserve-gib",type=float,default=8.)
    revised=commands.add_parser("requalify")
    revised.add_argument("dataset",type=Path);revised.add_argument("--output-dir",type=Path,required=True)
    revised.add_argument("--contract",type=Path)
    for name in ("calibrate","validate","summary","split","export"):
        command=commands.add_parser(name);command.add_argument("dataset",type=Path)
        if name=="validate":
            command.add_argument("--deep",action="store_true")
            command.add_argument("--report",type=Path,help="Write the audit outside an immutable dataset.")
        if name=="split":command.add_argument("--seed",type=int,default=1741)
        if name=="export":
            command.add_argument("--output-dir",type=Path,required=True)
            command.add_argument("--window-s",type=float,default=5.)
            command.add_argument("--stride-s",type=float,default=1.)
            command.add_argument("--max-gib",type=float,default=4.)
            command.add_argument("--reserve-gib",type=float,default=8.)
    args=parser.parse_args(argv)
    try:
        if args.command=="sources":
            from .sources import inventory
            result=inventory(args.probe_native)
        elif args.command=="generate":
            from .pipeline import generate
            result=generate(args.output_dir,count=args.count,positive=args.positive,negative=args.negative,
                max_candidates=args.max_candidates,workers=args.workers,max_seconds=args.max_seconds,
                max_gib=args.max_gib,reserve_gib=args.reserve_gib,phase=args.phase,contract_file=args.contract)
        elif args.command=="requalify":
            from .pipeline import requalify
            result=requalify(args.dataset,args.output_dir,contract_file=args.contract)
        elif args.command in ("calibrate","summary"):
            from .pipeline import calibrate,summary
            result=(calibrate if args.command=="calibrate" else summary)(args.dataset)
        else:
            from .export import validate,split,export
            if args.command=="validate":result=validate(args.dataset,deep=args.deep,report_path=args.report)
            elif args.command=="split":result=split(args.dataset,seed=args.seed)
            else:result=export(args.dataset,args.output_dir,window_s=args.window_s,stride_s=args.stride_s,
                              max_gib=args.max_gib,reserve_gib=args.reserve_gib)
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        if args.command=="validate" and not result["valid"]:
            return 1
        if args.command=="generate" and result["stop_reason"]!="quotas_reached":
            return 3
        return 0
    except (ValueError,OSError,ImportError) as error:
        print(f"Dataset refused: {error}")
        return 2


if __name__=="__main__":
    raise SystemExit(main())
