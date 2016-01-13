import sys
import argparse
import logging
import memn2n


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


def main():
    explain = None
    parser = argparse.ArgumentParser(
                description="MemN2N for Facebook bAbi project"
             )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
                "--debug", default=False,
                action="store_true",
                help="run in debugging mode."
    )
    group.add_argument(
                "-q", "--quiet", action="store_true"
            )
    parser.add_argument(
                "-e", "--explain", type=str, default=None,
                help="print the usage of argument json file ('' for all)"
                )
    args = parser.parse_args()

    model = memn2n.Model()
    if args.explain is not None:
        model.args.explain_args('rnn.example.json', arg=args.explain)
        sys.exit(1)

    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level,
                        format='[N2N %(levelname)s] %(message)s')
    model.train()

if __name__ == '__main__':
    main()
