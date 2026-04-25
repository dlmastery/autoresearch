# Pushing the fraud_ecommerce work to GitHub

The work is committed locally on branch `fraud-ecommerce-autoresearch`, against
the upstream `dlmastery/autoresearch`. To publish your version of the dashboard
and enable GitHub Pages serving, you need to push the branch to a fork or new
repo of your own (you cannot push to `dlmastery/autoresearch` without write
access).

## One-time setup

1. **Create your fork or new repo on GitHub**, e.g. `<your-username>/autoresearch`
   (a fork of dlmastery/autoresearch is easiest — keeps the upstream link).

2. **Add it as a git remote here** (run from `autoresearch/`):
   ```bash
   cd C:/Users/abhir/clauderesearch/autoresearch
   git remote add origin-mine https://github.com/<your-username>/autoresearch.git
   ```

3. **Push the branch:**
   ```bash
   git push -u origin-mine fraud-ecommerce-autoresearch
   ```

4. **(Optional) merge to main** on GitHub via Pull Request UI, or push directly:
   ```bash
   git checkout master
   git merge fraud-ecommerce-autoresearch
   git push origin-mine master
   ```

## Enable GitHub Pages

After pushing:

1. Go to your repo on GitHub → **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: `master` (or `main` — whichever you pushed to) — Folder: `/docs`
4. Save. Within 60 seconds your dashboard will be at:

   ```
   https://<your-username>.github.io/autoresearch/fraud_ecommerce/
   ```

The dashboard auto-refreshes every 30 seconds (browser-side fetch). To publish
new experiment results, simply:

```bash
python generalized_ml_autoresearch/examples/fraud_ecommerce/sync_dashboard.py
git add docs/fraud_ecommerce/
git commit -m "refresh dashboard"
git push origin-mine
```

GitHub Pages will pick up the change in ~30-60 seconds.

## Local serving

For development, the dashboard works fully offline:

```bash
python -m http.server 8765 --directory generalized_ml_autoresearch/examples/fraud_ecommerce/autoresearch_results
```

Open `http://localhost:8765/dashboard.html`.

## Verify everything

```bash
# Check the commit landed
git log --oneline -5

# Verify GitHub Pages files exist
ls docs/fraud_ecommerce/

# Verify the leaderboard
python -c "
import json
log = open('generalized_ml_autoresearch/examples/fraud_ecommerce/autoresearch_results/experiment_log.jsonl').read().strip().split('\n')
for line in log:
    d = json.loads(line)
    print(f\"  Exp {d['experiment_num']:<3} {d['backbone']:<10} test_auc={d['test_primary']:.4f}  {d['status']}\")
"
```
