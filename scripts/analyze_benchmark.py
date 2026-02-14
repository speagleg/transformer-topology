"""Parse benchmark training logs and produce summary tables.

Usage:
    python scripts/analyze_benchmark.py [data/benchmark_jobs]
"""

import re
import sys
from pathlib import Path


# Map job files to task/variant based on job list files
def _parse_job_list(path: Path) -> dict[int, dict]:
    """Parse a jobs.txt file into {job_id: {task, variant, results_file}}."""
    jobs = {}
    for i, line in enumerate(path.read_text().strip().splitlines()):
        m_task = re.search(r'--task\s+(\S+)', line)
        m_var = re.search(r'--variant\s+(\S+)', line)
        m_res = re.search(r'--results-file\s+(\S+)', line)
        if m_task and m_var:
            jobs[i] = {
                'task': m_task.group(1),
                'variant': m_var.group(1),
                'results_file': m_res.group(1) if m_res else None,
            }
    return jobs


def _parse_log(path: Path) -> dict:
    """Extract key metrics from a single job stdout log."""
    text = path.read_text()
    if not text.strip():
        return None  # Empty log

    result = {
        'epochs': [],
        'best_val': None,
        'best_epoch': None,
        'early_stopped': None,
        'id_accuracy': None,
        'id_val': None,
        'size_results': {},
        'topo_results': {},
        'per_class': None,
        'per_class_best': None,
        'params': None,
        'topo_sections': [],
    }

    # Parameters
    m = re.search(r'Parameters:\s+([\d,]+)', text)
    if m:
        result['params'] = int(m.group(1).replace(',', ''))

    # Split log into ID section and topo section
    topo_split = text.split('Training on topo-restricted split')
    id_text = topo_split[0]

    # Training epochs (ID section only)
    for m in re.finditer(
        r'Ep\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+val\s+([\d.]+)\s+\(([\d.]+)\)',
        id_text,
    ):
        epoch = int(m.group(1))
        train_loss = float(m.group(2))
        val_acc = float(m.group(3))
        val_loss = float(m.group(4))
        # Find the * marker — may be far from regex end in extended log format
        line_end = id_text.find('\n', m.end())
        if line_end == -1:
            line_end = len(id_text)
        is_best = '*' in id_text[m.end():line_end]
        result['epochs'].append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_acc': val_acc, 'val_loss': val_loss, 'is_best': is_best,
        })

    # Best val from ID training
    best_epochs = [e for e in result['epochs'] if e['is_best']]
    if best_epochs:
        result['best_val'] = best_epochs[-1]['val_acc']
        result['best_epoch'] = best_epochs[-1]['epoch']

    # Early stopping (ID section)
    m = re.search(r'Early stop at epoch (\d+)', id_text)
    if m:
        result['early_stopped'] = int(m.group(1))

    # ID accuracy
    m = re.search(r'ID accuracy:\s+([\d.]+)\s+\(val:\s+([\d.]+)', text)
    if m:
        result['id_accuracy'] = float(m.group(1))
        result['id_val'] = float(m.group(2))

    # Also try format with epoch info
    if not result['id_accuracy']:
        m = re.search(r'ID accuracy:\s+([\d.]+)', text)
        if m:
            result['id_accuracy'] = float(m.group(1))

    # Size results
    for m in re.finditer(r'Size n=(\d+):\s+([\d.]+)', text):
        n = int(m.group(1))
        acc = float(m.group(2))
        result['size_results'][n] = acc

    # Per-class accuracy
    m = re.search(r'Worst:\s+(.+)', text)
    if m:
        result['per_class'] = m.group(1).strip()
    m = re.search(r'Best:\s+(.+)', text)
    if m:
        result['per_class_best'] = m.group(1).strip()

    # Topo-transfer sections
    for i, section_text in enumerate(topo_split[1:]):
        topo = {'epochs': [], 'early_stopped': None, 'topo_accuracy': None, 'best_val': None}
        for m in re.finditer(
            r'Ep\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+val\s+([\d.]+)',
            section_text,
        ):
            topo['epochs'].append({
                'epoch': int(m.group(1)),
                'val_acc': float(m.group(3)),
            })
        m_stop = re.search(r'Early stop at epoch (\d+)', section_text)
        if m_stop:
            topo['early_stopped'] = int(m_stop.group(1))
        m_acc = re.search(r'Topo accuracy:\s+([\d.]+)', section_text)
        if m_acc:
            topo['topo_accuracy'] = float(m_acc.group(1))
        if topo['epochs']:
            topo['best_val'] = max(e['val_acc'] for e in topo['epochs'])
        result['topo_sections'].append(topo)

    return result


def analyze(base_dir: Path):
    """Analyze all benchmark logs and print summary."""
    # Collect all job results
    all_results = {}

    # Parse each launcher's job list and logs
    launchers = [
        ('Round 1', 'jobs.txt', 'logs'),
        ('Round 2', 'jobs_round2.txt', 'logs_round2'),
        ('Hodge Fix', 'jobs_hodge_fix.txt', 'logs_hodge_fix'),
        ('Hodge Quick', 'jobs_hodge_quick.txt', 'logs_hodge_quick'),
    ]

    for launcher_name, jobs_file, log_dir in launchers:
        jobs_path = base_dir / jobs_file
        logs_path = base_dir / log_dir
        if not jobs_path.exists() or not logs_path.exists():
            continue

        jobs = _parse_job_list(jobs_path)
        for job_id, job_info in jobs.items():
            log_file = logs_path / f'job_{job_id:04d}.stdout'
            if not log_file.exists():
                continue

            result = _parse_log(log_file)
            if result is None:
                continue  # Empty log

            key = (job_info['task'], job_info['variant'])

            # Skip symmetric (archived)
            if job_info['variant'] == 'symmetric':
                continue

            # Later launchers override earlier ones, unless the new result
            # has strictly less data (e.g. empty log vs completed job)
            if key in all_results:
                existing = all_results[key]
                new_has_data = bool(result['id_accuracy'] or result['epochs'])
                old_has_id = bool(existing['id_accuracy'])
                # Only keep old if it has ID accuracy and new has nothing
                if old_has_id and not new_has_data:
                    continue

            all_results[key] = {**job_info, **result, 'launcher': launcher_name}

    if not all_results:
        print("No results found. Check that logs exist in:", base_dir)
        return

    # Print summary tables
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group by task
    tasks = sorted(set(k[0] for k in all_results))
    variants = ['hierarchical', 'hierarchical_nowave']

    # ID Accuracy table
    print("\n## ID Accuracy (train & test n=20)\n")
    print(f"{'Task':<20} {'Hierarchical':>14} {'Hier-NoWave':>14} {'Delta':>8} {'Winner':>10}")
    print("-" * 70)
    for task in tasks:
        row = f"{task:<20}"
        accs = {}
        for var in variants:
            key = (task, var)
            if key in all_results:
                r = all_results[key]
                val = r['id_accuracy'] or r['best_val']
                if val is not None:
                    suffix = '' if r['early_stopped'] is not None or r['id_accuracy'] else '*'
                    accs[var] = val
                    row += f"  {val:>11.1%}{suffix}"
                else:
                    row += f"  {'training':>12}"
            else:
                row += f"  {'—':>12}"
        if len(accs) == 2:
            delta = accs.get('hierarchical', 0) - accs.get('hierarchical_nowave', 0)
            winner = 'wave' if delta > 0.01 else ('no-wave' if delta < -0.01 else 'tied')
            row += f"  {delta:>+6.1%}  {winner:>8}"
        print(row)

    # Size generalization table
    print("\n## Size Generalization\n")
    print(f"{'Task':<20} {'Variant':<12} {'n=20':>8} {'n=40':>8} {'n=80':>8} {'Drop 2x':>8} {'Drop 4x':>8}")
    print("-" * 80)
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in all_results:
                continue
            r = all_results[key]
            id_acc = r['id_accuracy'] or r['best_val']
            n40 = r['size_results'].get(40)
            n80 = r['size_results'].get(80)
            short_var = 'hier' if var == 'hierarchical' else 'nowave'
            row = f"{task:<20} {short_var:<12}"
            row += f"  {id_acc:>6.1%}" if id_acc is not None else f"  {'—':>6}"
            row += f"  {n40:>6.1%}" if n40 is not None else f"  {'—':>6}"
            row += f"  {n80:>6.1%}" if n80 is not None else f"  {'—':>6}"
            if id_acc is not None and n40 is not None:
                row += f"  {id_acc - n40:>+6.1%}"
            else:
                row += f"  {'':>8}"
            if id_acc is not None and n80 is not None:
                row += f"  {id_acc - n80:>+6.1%}"
            print(row)

    # Topo-transfer table
    print("\n## Topology Transfer\n")
    print(f"{'Task':<20} {'Variant':<12} {'Topo Acc':>10} {'Topo Val':>10} {'Topo Ep':>8} {'Status':<15}")
    print("-" * 78)
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in all_results:
                continue
            r = all_results[key]
            short_var = 'hier' if var == 'hierarchical' else 'nowave'
            if not r['topo_sections']:
                print(f"{task:<20} {short_var:<12}  {'—':>8}  {'—':>8}  {'—':>6}  {'not started':<15}")
                continue
            topo = r['topo_sections'][-1]  # Latest topo section
            n_ep = len(topo['epochs'])
            topo_acc = topo.get('topo_accuracy')
            best_val = topo.get('best_val')
            if topo['early_stopped'] is not None:
                status = f"stopped@{topo['early_stopped']}"
            elif topo_acc is not None:
                status = "done"
            else:
                status = f"training (ep {n_ep})"
            row = f"{task:<20} {short_var:<12}"
            row += f"  {topo_acc:>8.1%}" if topo_acc is not None else f"  {'—':>8}"
            row += f"  {best_val:>8.1%}" if best_val is not None else f"  {'—':>8}"
            row += f"  {n_ep:>6}"
            row += f"  {status:<15}"
            print(row)

    # Training summary
    print("\n## Training Summary\n")
    print(f"{'Task':<20} {'Variant':<12} {'Params':>8} {'Epochs':>8} {'Best Val':>10} {'Status':<20} {'Launcher':<15}")
    print("-" * 95)
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in all_results:
                continue
            r = all_results[key]
            short_var = 'hier' if var == 'hierarchical' else 'nowave'
            n_epochs = len(r['epochs'])
            best = r['best_val']
            params = r['params']
            if r['id_accuracy'] is not None and r['topo_sections']:
                topo = r['topo_sections'][-1]
                if topo.get('topo_accuracy') is not None:
                    status = "fully done"
                else:
                    status = f"topo ep {len(topo['epochs'])}"
            elif r['id_accuracy'] is not None:
                status = "ID done, eval OOD"
            elif r['early_stopped'] is not None:
                status = f"stopped@{r['early_stopped']}"
            elif n_epochs > 0:
                status = f"training (ep {n_epochs - 1})"
            else:
                status = "not started"
            row = f"{task:<20} {short_var:<12}"
            row += f"  {params:>6,}" if params else f"  {'—':>6}"
            row += f"  {n_epochs:>6}"
            row += f"  {best:>8.1%}" if best is not None else f"  {'—':>8}"
            row += f"  {status:<20}"
            row += f"  {r['launcher']:<15}"
            print(row)

    # Per-class accuracy (where available)
    has_per_class = False
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key in all_results and all_results[key]['per_class']:
                has_per_class = True
                break
    if has_per_class:
        print("\n## Per-Class Accuracy\n")
        for task in tasks:
            for var in variants:
                key = (task, var)
                if key not in all_results:
                    continue
                r = all_results[key]
                if r['per_class']:
                    short_var = 'hier' if var == 'hierarchical' else 'nowave'
                    print(f"  {task}/{short_var}:")
                    print(f"    Worst: {r['per_class']}")
                    if r['per_class_best']:
                        print(f"    Best:  {r['per_class_best']}")

    # Wave vs No-Wave summary
    print("\n## Wave Dynamics Impact Summary\n")
    print(f"{'Task':<20} {'Hier ID':>10} {'NoWave ID':>10} {'Delta':>8} {'Verdict':<30}")
    print("-" * 80)
    for task in tasks:
        h_key = (task, 'hierarchical')
        n_key = (task, 'hierarchical_nowave')
        h_acc = None
        n_acc = None
        if h_key in all_results:
            r = all_results[h_key]
            h_acc = r['id_accuracy'] or r['best_val']
        if n_key in all_results:
            r = all_results[n_key]
            n_acc = r['id_accuracy'] or r['best_val']
        row = f"{task:<20}"
        row += f"  {h_acc:>8.1%}" if h_acc is not None else f"  {'—':>8}"
        row += f"  {n_acc:>8.1%}" if n_acc is not None else f"  {'—':>8}"
        if h_acc is not None and n_acc is not None:
            delta = h_acc - n_acc
            if delta > 0.02:
                verdict = f"wave helps (+{delta:.1%})"
            elif delta < -0.02:
                verdict = f"wave hurts ({delta:+.1%})"
            else:
                verdict = f"negligible ({delta:+.1%})"
            row += f"  {delta:>+6.1%}  {verdict:<30}"
        print(row)

    print("\n" + "=" * 80)


if __name__ == '__main__':
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/benchmark_jobs')
    analyze(base)
