# AutoResearch Documentation Index

This directory contains all project documentation, organized by topic area.

## architecture/

System design, architecture decisions, and configuration.

- [project-overview.md](architecture/project-overview.md) -- Project charter and overview for the AutoResearch FX prediction system.
- [system-design.md](architecture/system-design.md) -- High-level system architecture, component design, and data flow.
- [model-architecture.md](architecture/model-architecture.md) -- Neural network backbone architectures and adapter design.
- [configuration-management.md](architecture/configuration-management.md) -- Configuration management, version control, and branch strategy.

## data/

Data engineering, features, and requirements.

- [data-engineering.md](data/data-engineering.md) -- Data pipeline: download, caching, feature engineering, and preprocessing.
- [requirements-specification.md](data/requirements-specification.md) -- SWEBoK-aligned requirements specification for the system.

## training/

Training infrastructure and hyperparameter optimization.

- [training-infrastructure.md](training/training-infrastructure.md) -- Training loop, loss functions, early stopping, and GPU utilization.
- [hyperparameter-sweeps.md](training/hyperparameter-sweeps.md) -- Hyperparameter sweep strategy, tuning methodology, and results.
- [hyperparameter-research.md](training/hyperparameter-research.md) -- LFM2.5-350M-specific hyperparameter research from technical reports.

## evaluation/

Metrics, testing, and quality assurance.

- [evaluation-framework.md](evaluation/evaluation-framework.md) -- Evaluation metrics: Sharpe, PSR, DSR, IC, and trading report methodology.
- [testing-strategy.md](evaluation/testing-strategy.md) -- Unit, integration, and system testing strategy.
- [quality-best-practices.md](evaluation/quality-best-practices.md) -- Quality assurance guidelines and best practices.

## research/

Literature surveys, comparisons, and retrospectives.

- [arxiv-sota-survey.md](research/arxiv-sota-survey.md) -- ArXiv survey of state-of-the-art time series and FX prediction methods (2024-2026).
- [autoresearch-comparison.md](research/autoresearch-comparison.md) -- Comparison with Karpathy's autoresearch and the autonomous ML research landscape.
- [research-retrospective.md](research/research-retrospective.md) -- Brainstorming and research retrospective from the initial currency prediction design session.
- [retrospective.md](research/retrospective.md) -- Multi-session development retrospective covering architecture and implementation evolution.

## operations/

Deployment, MLOps, and autonomous optimization.

- [operations-deployment.md](operations/operations-deployment.md) -- Operations, deployment, and maintenance procedures.
- [autonomous-optimization.md](operations/autonomous-optimization.md) -- Claude API agent loop for autonomous experiment optimization.
- [implementation-status.md](operations/implementation-status.md) -- Implementation status tracker and progress dashboard.

## plans/

Design and implementation plans.

- [2026-04-04-currency-prediction-design.md](plans/2026-04-04-currency-prediction-design.md) -- Original design document for currency prediction with LFM2.5 and autoresearch optimizer.
- [2026-04-04-currency-prediction-impl.md](plans/2026-04-04-currency-prediction-impl.md) -- Detailed implementation plan for the currency prediction system.
