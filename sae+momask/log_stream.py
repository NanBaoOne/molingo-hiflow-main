"""Run a command with bounded logs; forward dynamic output only to terminals."""
import argparse
from collections import deque
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', required=True)
    parser.add_argument('--max-mb', type=int, default=20)
    parser.add_argument('--backups', type=int, default=2)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.max_mb < 1 or args.backups < 0:
        parser.error('command, positive max-mb and nonnegative backups required')
    Path(args.file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('bounded-run')
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(args.file, maxBytes=args.max_mb*1024*1024,
                                  backupCount=args.backups, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    terminal = sys.stdout.isatty()
    env = dict(os.environ, MOMASK_TTY='1' if terminal else '0')
    print(f'Log: {Path(args.file).resolve()} (rotation: {args.max_mb} MiB x {args.backups+1})', flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    # Keep dynamic CR updates on the terminal, never persist tqdm refreshes.
    pending = b''
    recent = deque(maxlen=80)
    ansi = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
    def record(raw):
        line = ansi.sub('', raw.decode('utf-8', errors='replace')).strip()
        if line and not re.search(r'\d+%\|', line):
            recent.append(line[:4096])
            if line.startswith(('EVAL_METRICS ', 'VALIDATION_LOSS ')):
                logger.info(line)
    try:
        while True:
            chunk = os.read(process.stdout.fileno(), 8192)
            if not chunk:
                break
            if terminal:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            pending += chunk
            parts = re.split(b'[\r\n]', pending)
            pending = parts.pop()
            for raw in parts:
                record(raw)
            # Guard against a dependency emitting a huge line without delimiters.
            if len(pending) > 65536:
                record(pending[:65536])
                pending = pending[65536:]
        if pending.strip():
            record(pending)
        code = process.wait()
        if code:
            # Diagnostics are separate from metric/loss logs, and bounded too.
            errors = RotatingFileHandler(args.file+'.error.log', maxBytes=args.max_mb*1024*1024,
                                         backupCount=args.backups, encoding='utf-8')
            errors.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            for line in recent:
                errors.emit(logging.LogRecord('error', logging.ERROR, __file__, 0, line, (), None))
            errors.close()
            print(f'Command failed (exit {code}); diagnostics: {args.file}.error.log', file=sys.stderr, flush=True)
        return code
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        return 130
    finally:
        handler.close()


if __name__ == '__main__':
    sys.exit(main())
