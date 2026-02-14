"""Deep analysis of benchmark training logs.

Extracts detailed metrics: learning curves, convergence rates, loss dynamics,
per-class breakdowns, efficiency metrics, and cross-task comparisons.

Usage:
    python scripts/deep_analysis.py [data/benchmark_jobs]
"""

import re
import sys
import math
from pathlib import Path
from collections import defaultdict


def _parse_job_list(path: Path) -> dict[int, dict]:
    jobs = {}
    for i, line in enumerate(path.read_text().strip().splitlines()):
        m_task = re.search(r'--task\s+(\S+)', line)
        m_var = re.search(r'--variant\s+(\S+)', line)
        if m_task and m_var:
            jobs[i] = {'task': m_task.group(1), 'variant': m_var.group(1)}
    return jobs


def _parse_full_log(path: Path) -> dict | None:
    text = path.read_text()
    if not text.strip():
        return None

    result = {
        'params': None,
        'id_epochs': [],
        'id_accuracy': None,
        'id_val': None,
        'id_best_epoch': None,
        'id_total_epochs': None,
        'early_stopped': None,
        'size_results': {},
        'per_class_worst': None,
        'per_class_best': None,
        'topo_epochs': [],
        'topo_accuracy': None,
        'topo_early_stopped': None,
        'seed': None,
    }

    # Seed
    m = re.search(r'seed=(\d+)', text)
    if m:
        result['seed'] = int(m.group(1))

    # Parameters
    m = re.search(r'Parameters:\s+([\d,]+)', text)
    if m:
        result['params'] = int(m.group(1).replace(',', ''))

    # Split into ID and topo sections
    topo_split = text.split('Training on topo-restricted split')
    id_text = topo_split[0]
    topo_text = topo_split[1] if len(topo_split) > 1 else ''

    # ID epochs - handle both formats (with and without gn/lr)
    for m in re.finditer(
        r'Ep\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+val\s+([\d.]+)\s+\(([\d.]+)\)',
        id_text,
    ):
        line_end = id_text.find('\n', m.end())
        if line_end == -1:
            line_end = len(id_text)
        line_rest = id_text[m.end():line_end]
        is_best = '*' in line_rest

        # Extract gradient norm and lr if present
        gn = None
        lr = None
        m_gn = re.search(r'gn\s+([\d.]+)', line_rest)
        m_lr = re.search(r'lr\s+([\d.e+-]+)', line_rest)
        m_time = re.search(r'(\d+)s', line_rest)
        if m_gn:
            gn = float(m_gn.group(1))
        if m_lr:
            lr = float(m_lr.group(1))

        result['id_epochs'].append({
            'epoch': int(m.group(1)),
            'train_loss': float(m.group(2)),
            'val_acc': float(m.group(3)),
            'val_loss': float(m.group(4)),
            'is_best': is_best,
            'grad_norm': gn,
            'lr': lr,
            'time_s': int(m_time.group(1)) if m_time else None,
        })

    # Early stopping (ID)
    m = re.search(r'Early stop at epoch (\d+)', id_text)
    if m:
        result['early_stopped'] = int(m.group(1))
        result['id_total_epochs'] = int(m.group(1)) + 1

    # ID accuracy
    m = re.search(r'ID accuracy:\s+([\d.]+)\s+\(val:\s+([\d.]+)', text)
    if m:
        result['id_accuracy'] = float(m.group(1))
        result['id_val'] = float(m.group(2))
    m2 = re.search(r'ep:\s+(\d+)/(\d+)', text)
    if m2:
        result['id_best_epoch'] = int(m2.group(1))

    # Size results
    for m in re.finditer(r'Size n=(\d+):\s+([\d.]+)', text):
        result['size_results'][int(m.group(1))] = float(m.group(2))

    # Per-class
    m = re.search(r'Worst:\s+(.+)', text)
    if m:
        result['per_class_worst'] = m.group(1).strip()
    m = re.search(r'Best:\s+(.+)', text)
    if m:
        result['per_class_best'] = m.group(1).strip()

    # Topo epochs
    for m in re.finditer(
        r'Ep\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+val\s+([\d.]+)',
        topo_text,
    ):
        line_end = topo_text.find('\n', m.end())
        if line_end == -1:
            line_end = len(topo_text)
        line_rest = topo_text[m.end():line_end]
        is_best = '*' in line_rest
        result['topo_epochs'].append({
            'epoch': int(m.group(1)),
            'val_acc': float(m.group(3)),
            'is_best': is_best,
        })

    # Topo results
    m = re.search(r'Topo transfer:\s+([\d.]+)', text)
    if m:
        result['topo_accuracy'] = float(m.group(1))
    m = re.search(r'Early stop at epoch (\d+)', topo_text)
    if m:
        result['topo_early_stopped'] = int(m.group(1))

    return result


def _load_all(base_dir: Path) -> dict:
    launchers = [
        ('Round 1', 'jobs.txt', 'logs'),
        ('Round 2', 'jobs_round2.txt', 'logs_round2'),
        ('Hodge Fix', 'jobs_hodge_fix.txt', 'logs_hodge_fix'),
        ('Hodge Quick', 'jobs_hodge_quick.txt', 'logs_hodge_quick'),
    ]
    all_results = {}
    for launcher_name, jobs_file, log_dir in launchers:
        jobs_path = base_dir / jobs_file
        logs_path = base_dir / log_dir
        if not jobs_path.exists() or not logs_path.exists():
            continue
        jobs = _parse_job_list(jobs_path)
        for job_id, job_info in jobs.items():
            if job_info['variant'] == 'symmetric':
                continue
            log_file = logs_path / f'job_{job_id:04d}.stdout'
            if not log_file.exists():
                continue
            result = _parse_full_log(log_file)
            if result is None:
                continue
            key = (job_info['task'], job_info['variant'])
            if key in all_results:
                existing = all_results[key]
                if existing['id_accuracy'] and not result['id_accuracy'] and not result['id_epochs']:
                    continue
            all_results[key] = {**job_info, **result, 'launcher': launcher_name}
    return all_results


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def analyze(base_dir: Path):
    data = _load_all(base_dir)
    if not data:
        print("No data found.")
        return

    tasks = sorted(set(k[0] for k in data))
    variants = ['hierarchical', 'hierarchical_nowave']

    # =========================================================================
    # SECTION 1: ID Accuracy Deep Dive
    # =========================================================================
    print_section("1. IN-DISTRIBUTION ACCURACY")

    print(f"\n{'Task':<18} {'Variant':<10} {'ID Acc':>8} {'Val Acc':>8} {'Best Ep':>8} "
          f"{'Tot Ep':>8} {'Params':>8} {'Seed':>6}")
    print("-" * 88)
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            short = 'hier' if var == 'hierarchical' else 'nowave'
            id_acc = r['id_accuracy']
            id_val = r['id_val']
            best_ep = r['id_best_epoch']
            tot_ep = r['id_total_epochs'] or len(r['id_epochs'])
            params = r['params']
            seed = r['seed']
            print(f"{task:<18} {short:<10} "
                  f"{id_acc:>7.1%} " if id_acc else f"{task:<18} {short:<10} {'—':>7} ", end="")
            if not id_acc:
                best_epochs = [e for e in r['id_epochs'] if e['is_best']]
                best_val = best_epochs[-1]['val_acc'] if best_epochs else None
                print(f" {best_val:>7.1%}*" if best_val else f" {'—':>7}", end="")
            else:
                print(f" {id_val:>7.1%}" if id_val else f" {'—':>7}", end="")
            print(f" {best_ep:>7}" if best_ep is not None else f" {'—':>7}", end="")
            print(f" {tot_ep:>7}" if tot_ep else f" {'—':>7}", end="")
            print(f" {params:>7,}" if params else f" {'—':>7}", end="")
            print(f" {seed:>5}" if seed else f" {'—':>5}")

    # Random baselines
    print(f"\n  Random baselines: bfs ~6.3% (1/16), diverse ~9.1% (1/11),")
    print(f"  spectral_gap ~12.5% (1/8), hodge_class ~33.3% (1/3), prop_delay ~6.3% (1/16)")

    # =========================================================================
    # SECTION 2: Learning Dynamics
    # =========================================================================
    print_section("2. LEARNING DYNAMICS")

    for task in tasks:
        print(f"\n--- {task} ---")
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            epochs = r['id_epochs']
            if not epochs:
                continue
            short = 'hier' if var == 'hierarchical' else 'nowave'

            # Key metrics
            losses = [e['train_loss'] for e in epochs]
            val_accs = [e['val_acc'] for e in epochs]
            val_losses = [e['val_loss'] for e in epochs]
            times = [e['time_s'] for e in epochs if e['time_s']]
            grad_norms = [e['grad_norm'] for e in epochs if e['grad_norm'] is not None]

            # Convergence speed: epoch where val acc first exceeds 90% of final
            best_val = max(val_accs)
            threshold_90 = best_val * 0.9
            first_90 = next((e['epoch'] for e in epochs if e['val_acc'] >= threshold_90), None)

            # Overfitting: gap between train loss and val loss at end
            final_train_loss = losses[-1]
            final_val_loss = val_losses[-1]
            overfit_gap = final_val_loss - final_train_loss

            # Loss reduction rate (first 5 epochs)
            if len(losses) >= 5:
                early_loss_drop = losses[0] - losses[4]
            else:
                early_loss_drop = losses[0] - losses[-1]

            print(f"\n  {short}:")
            print(f"    Epochs: {len(epochs)}, Best val: {best_val:.1%} (ep {epochs[val_accs.index(best_val)]['epoch']})")
            print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f} (Δ={losses[0]-losses[-1]:.4f})")
            print(f"    Val loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
            print(f"    Overfit gap (val_loss - train_loss): {overfit_gap:+.4f}"
                  f" {'⚠ overfitting' if overfit_gap > 0.2 else ''}")
            print(f"    Early loss drop (ep 0→4): {early_loss_drop:.4f}")
            if first_90 is not None:
                print(f"    Reached 90% of best val ({threshold_90:.1%}) at epoch {first_90}")
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                print(f"    Time/epoch: {avg_time:.0f}s avg, {total_time/3600:.1f}h total")
            if grad_norms:
                print(f"    Grad norm: {grad_norms[0]:.2f} → {grad_norms[-1]:.2f}"
                      f" (max {max(grad_norms):.2f})")

    # =========================================================================
    # SECTION 3: Size Generalization Analysis
    # =========================================================================
    print_section("3. SIZE GENERALIZATION")

    print(f"\n{'Task':<18} {'Variant':<10} {'n=20':>8} {'n=40':>8} {'n=80':>8} "
          f"{'Drop 2x':>8} {'Drop 4x':>8} {'Retain 2x':>10} {'Retain 4x':>10}")
    print("-" * 100)
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            id_acc = r['id_accuracy']
            if not id_acc:
                best_epochs = [e for e in r['id_epochs'] if e['is_best']]
                id_acc = best_epochs[-1]['val_acc'] if best_epochs else None
            n40 = r['size_results'].get(40)
            n80 = r['size_results'].get(80)
            short = 'hier' if var == 'hierarchical' else 'nowave'
            row = f"{task:<18} {short:<10}"
            row += f" {id_acc:>7.1%}" if id_acc else f" {'—':>7}"
            row += f" {n40:>7.1%}" if n40 is not None else f" {'—':>7}"
            row += f" {n80:>7.1%}" if n80 is not None else f" {'—':>7}"
            if id_acc and n40:
                drop2 = id_acc - n40
                retain2 = n40 / id_acc
                row += f" {drop2:>+7.1%} {retain2:>9.1%}"
            else:
                row += f" {'':>8} {'':>10}"
            if id_acc and n80:
                drop4 = id_acc - n80
                retain4 = n80 / id_acc
                row += f" {drop4:>+7.1%} {retain4:>9.1%}"
            print(row)

    # Size gen ranking
    print("\n  Size Generalization Ranking (by retention at largest tested size):")
    rankings = []
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            id_acc = r['id_accuracy']
            if not id_acc:
                best_epochs = [e for e in r['id_epochs'] if e['is_best']]
                id_acc = best_epochs[-1]['val_acc'] if best_epochs else None
            if not id_acc:
                continue
            sizes = r['size_results']
            if not sizes:
                continue
            max_size = max(sizes.keys())
            ood_acc = sizes[max_size]
            retain = ood_acc / id_acc
            short = 'hier' if var == 'hierarchical' else 'nowave'
            rankings.append((retain, task, short, id_acc, ood_acc, max_size))
    rankings.sort(reverse=True)
    for i, (retain, task, var, id_acc, ood_acc, n) in enumerate(rankings, 1):
        print(f"    {i}. {task}/{var}: {retain:.1%} retained at n={n}"
              f" ({id_acc:.1%} → {ood_acc:.1%})")

    # =========================================================================
    # SECTION 4: Wave Dynamics Impact
    # =========================================================================
    print_section("4. WAVE DYNAMICS IMPACT")

    print(f"\n{'Task':<18} {'Hier ID':>8} {'NW ID':>8} {'Δ ID':>8} "
          f"{'Hier n40':>8} {'NW n40':>8} {'Δ n40':>8} "
          f"{'Hier n80':>8} {'NW n80':>8} {'Δ n80':>8}")
    print("-" * 100)
    for task in tasks:
        h = data.get((task, 'hierarchical'))
        n = data.get((task, 'hierarchical_nowave'))
        if not h or not n:
            continue

        def get_acc(r):
            if r['id_accuracy']:
                return r['id_accuracy']
            best = [e for e in r['id_epochs'] if e['is_best']]
            return best[-1]['val_acc'] if best else None

        h_id = get_acc(h)
        n_id = get_acc(n)
        h_40 = h['size_results'].get(40)
        n_40 = n['size_results'].get(40)
        h_80 = h['size_results'].get(80)
        n_80 = n['size_results'].get(80)

        row = f"{task:<18}"
        row += f" {h_id:>7.1%}" if h_id else f" {'—':>7}"
        row += f" {n_id:>7.1%}" if n_id else f" {'—':>7}"
        if h_id and n_id:
            row += f" {h_id-n_id:>+7.1%}"
        else:
            row += f" {'—':>7}"
        row += f" {h_40:>7.1%}" if h_40 else f" {'—':>7}"
        row += f" {n_40:>7.1%}" if n_40 else f" {'—':>7}"
        if h_40 is not None and n_40 is not None:
            row += f" {h_40-n_40:>+7.1%}"
        else:
            row += f" {'—':>7}"
        row += f" {h_80:>7.1%}" if h_80 else f" {'—':>7}"
        row += f" {n_80:>7.1%}" if n_80 else f" {'—':>7}"
        if h_80 is not None and n_80 is not None:
            row += f" {h_80-n_80:>+7.1%}"
        else:
            row += f" {'—':>7}"
        print(row)

    # Wave impact summary
    print("\n  Wave Impact Verdict:")
    for task in tasks:
        h = data.get((task, 'hierarchical'))
        n = data.get((task, 'hierarchical_nowave'))
        if not h or not n:
            continue
        def get_acc(r):
            if r['id_accuracy']:
                return r['id_accuracy']
            best = [e for e in r['id_epochs'] if e['is_best']]
            return best[-1]['val_acc'] if best else None
        h_id = get_acc(h)
        n_id = get_acc(n)
        if h_id is None or n_id is None:
            continue
        delta = h_id - n_id

        # Also check size gen
        h_80 = h['size_results'].get(80)
        n_80 = n['size_results'].get(80)
        h_40 = h['size_results'].get(40)
        n_40 = n['size_results'].get(40)

        verdict_parts = []
        if delta > 0.02:
            verdict_parts.append(f"ID: wave helps (+{delta:.1%})")
        elif delta < -0.02:
            verdict_parts.append(f"ID: wave hurts ({delta:+.1%})")
        else:
            verdict_parts.append(f"ID: negligible ({delta:+.1%})")

        if h_40 is not None and n_40 is not None:
            d40 = h_40 - n_40
            if abs(d40) > 0.02:
                verdict_parts.append(f"n=40: {'wave helps' if d40>0 else 'wave hurts'} ({d40:+.1%})")

        if h_80 is not None and n_80 is not None:
            d80 = h_80 - n_80
            if abs(d80) > 0.02:
                verdict_parts.append(f"n=80: {'wave helps' if d80>0 else 'wave hurts'} ({d80:+.1%})")

        print(f"    {task}: {'; '.join(verdict_parts)}")

    # =========================================================================
    # SECTION 5: Topology Transfer
    # =========================================================================
    print_section("5. TOPOLOGY TRANSFER")

    print(f"\n{'Task':<18} {'Variant':<10} {'ID Acc':>8} {'Topo Acc':>9} {'Topo Val':>9} "
          f"{'Topo Ep':>8} {'Transfer':>9} {'Status':<15}")
    print("-" * 98)
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            short = 'hier' if var == 'hierarchical' else 'nowave'
            id_acc = r['id_accuracy']
            if not id_acc:
                best = [e for e in r['id_epochs'] if e['is_best']]
                id_acc = best[-1]['val_acc'] if best else None

            topo_acc = r['topo_accuracy']
            topo_eps = r['topo_epochs']
            topo_best_val = max((e['val_acc'] for e in topo_eps), default=None)
            n_topo = len(topo_eps)

            # Transfer ratio
            transfer = None
            if topo_acc and id_acc:
                transfer = topo_acc / id_acc
            elif topo_best_val and id_acc:
                transfer = topo_best_val / id_acc

            if r['topo_accuracy'] is not None:
                status = "done"
            elif r['topo_early_stopped'] is not None:
                status = f"stopped@{r['topo_early_stopped']}"
            elif n_topo > 0:
                status = f"ep {n_topo}"
            else:
                status = "not started"

            row = f"{task:<18} {short:<10}"
            row += f" {id_acc:>7.1%}" if id_acc else f" {'—':>7}"
            row += f" {topo_acc:>8.1%}" if topo_acc is not None else f" {'—':>8}"
            row += f" {topo_best_val:>8.1%}" if topo_best_val is not None else f" {'—':>8}"
            row += f" {n_topo:>7}"
            row += f" {transfer:>8.1%}" if transfer is not None else f" {'—':>8}"
            row += f"  {status:<15}"
            print(row)

    # =========================================================================
    # SECTION 6: Per-Class Analysis
    # =========================================================================
    print_section("6. PER-CLASS ACCURACY")

    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            if not r['per_class_worst']:
                continue
            short = 'hier' if var == 'hierarchical' else 'nowave'
            print(f"\n  {task}/{short}:")
            print(f"    Worst: {r['per_class_worst']}")
            if r['per_class_best']:
                print(f"    Best:  {r['per_class_best']}")

            # Parse per-class to find zero-accuracy classes
            zeros = re.findall(r'(\w+)=0%\((\d+)\)', r['per_class_worst'])
            if zeros:
                total_zero_samples = sum(int(n) for _, n in zeros)
                print(f"    Zero-accuracy classes: {len(zeros)}, "
                      f"affecting {total_zero_samples} test samples")

    # =========================================================================
    # SECTION 7: Training Efficiency
    # =========================================================================
    print_section("7. TRAINING EFFICIENCY")

    print(f"\n{'Task':<18} {'Variant':<10} {'Epochs':>7} {'Time/Ep':>8} {'Total':>8} "
          f"{'Final Loss':>10} {'Loss Drop':>10} {'Converge':>9}")
    print("-" * 90)
    for task in tasks:
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            epochs = r['id_epochs']
            if not epochs:
                continue
            short = 'hier' if var == 'hierarchical' else 'nowave'
            n_ep = len(epochs)
            times = [e['time_s'] for e in epochs if e['time_s']]
            avg_time = sum(times) / len(times) if times else 0
            total_s = sum(times) if times else 0
            init_loss = epochs[0]['train_loss']
            final_loss = epochs[-1]['train_loss']
            loss_drop = init_loss - final_loss

            # Convergence: epochs to reach 95% of best val
            val_accs = [e['val_acc'] for e in epochs]
            best_val = max(val_accs)
            thresh95 = best_val * 0.95
            converge_ep = next((e['epoch'] for e in epochs if e['val_acc'] >= thresh95), None)

            print(f"{task:<18} {short:<10} {n_ep:>6} {avg_time:>7.0f}s "
                  f"{total_s/3600:>7.1f}h {final_loss:>9.4f} {loss_drop:>+9.4f} "
                  f"{'ep '+str(converge_ep) if converge_ep is not None else '—':>8}")

    # =========================================================================
    # SECTION 8: Cross-Task Summary Matrix
    # =========================================================================
    print_section("8. CROSS-TASK SUMMARY")

    print("\n  Task Properties vs Performance:")
    print(f"\n  {'Task':<18} {'Type':<20} {'ID Best':>8} {'Size Gen':>10} {'Topo Gen':>10} {'Wave?':>8}")
    print("  " + "-" * 78)

    task_types = {
        'bfs': 'distance classification',
        'diverse': 'multi-hop reachability',
        'hodge_class': 'spectral decomposition',
        'propagation_delay': 'edge-weighted path',
        'spectral_gap': 'global spectral prop',
    }

    for task in tasks:
        ttype = task_types.get(task, '?')
        # Best ID across variants
        best_id = 0
        best_var = ''
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            acc = r['id_accuracy']
            if not acc:
                best = [e for e in r['id_epochs'] if e['is_best']]
                acc = best[-1]['val_acc'] if best else 0
            if acc and acc > best_id:
                best_id = acc
                best_var = 'hier' if var == 'hierarchical' else 'nw'

        # Best size gen (retention ratio)
        best_retain = 0
        retain_label = '—'
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            acc = r['id_accuracy']
            if not acc:
                best = [e for e in r['id_epochs'] if e['is_best']]
                acc = best[-1]['val_acc'] if best else None
            if not acc:
                continue
            for n, ood in r['size_results'].items():
                retain = ood / acc
                if retain > best_retain:
                    best_retain = retain
                    short = 'hier' if var == 'hierarchical' else 'nw'
                    retain_label = f"{retain:.0%} ({short})"

        # Topo gen
        topo_label = '—'
        for var in variants:
            key = (task, var)
            if key not in data:
                continue
            r = data[key]
            if r['topo_accuracy'] is not None:
                topo_label = f"{r['topo_accuracy']:.0%}"
                break
            elif r['topo_epochs']:
                best_topo = max(e['val_acc'] for e in r['topo_epochs'])
                topo_label = f"~{best_topo:.0%}*"

        # Wave benefit
        h = data.get((task, 'hierarchical'))
        n = data.get((task, 'hierarchical_nowave'))
        wave_label = '—'
        if h and n:
            def ga(r):
                if r['id_accuracy']:
                    return r['id_accuracy']
                best = [e for e in r['id_epochs'] if e['is_best']]
                return best[-1]['val_acc'] if best else None
            h_a = ga(h)
            n_a = ga(n)
            if h_a and n_a:
                d = h_a - n_a
                wave_label = f"{d:+.1%}" if abs(d) > 0.01 else "~0"

        print(f"  {task:<18} {ttype:<20} {best_id:>7.1%} {retain_label:>10} "
              f"{topo_label:>10} {wave_label:>8}")

    # =========================================================================
    # SECTION 9: Key Findings
    # =========================================================================
    print_section("9. KEY FINDINGS")

    print("""
  1. SIZE GENERALIZATION
     - prop_delay/nowave retains 86.4% accuracy at 4x size (best by far)
     - All other tasks collapse to 20-36% at 2x and 16-24% at 4x
     - Edge-local tasks generalize; global/spectral tasks do not
     - Wave dynamics consistently worsen size generalization

  2. WAVE DYNAMICS
     - Net negative: helps 1 task (diverse +2.7%), hurts 1 (prop_delay -5.5%)
     - On size generalization: always equal or worse than no-wave
     - Root cause: spectral features (eigenvalues) are size-dependent
     - The learned diffusion_time/wave_damping don't transfer

  3. IN-DISTRIBUTION LEARNING
     - Architecture learns effectively across all task types
     - BFS: 99%+ (near perfect), Diverse: 89-93%, Spectral: 82-83%
     - Hodge: 55% at epoch 4 (not converged, likely 60-70% final)
     - ~162K params is sufficient for n=20 graphs

  4. TOPOLOGY TRANSFER
     - BFS transfers well (~99% topo val) — algorithm is topology-agnostic
     - Spectral_gap fails completely (11.8% ≈ random) — expected
     - Diverse/prop_delay show moderate transfer (50-95% topo val)

  5. ARCHITECTURE IMPLICATIONS
     - hierarchical_nowave should be the default variant
     - Wave module needs size-invariant redesign before being useful
     - The GNN executive + TAT pattern is validated for learning
     - Size generalization is the critical unsolved problem
""")

    print("=" * 80)


if __name__ == '__main__':
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/benchmark_jobs')
    analyze(base)
