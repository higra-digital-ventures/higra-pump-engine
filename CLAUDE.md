# HPE — Higra Pump Engine (Estado Atual)

> Atualizar este arquivo ao final de cada sessao de desenvolvimento.
> Este e o primeiro documento que o Claude Code deve ler antes de qualquer tarefa.

---

## Estado Geral

- **Data**: Junho 2026
- **Escopo**: Plataforma de turbomaquinas hidraulicas — **bombas, turbinas Francis e pump-turbines** (nao apenas bombas, apesar do nome). Inclui tambem axiais, inducers, ventiladores (sirocco/axial fan) e turbina radial.
- **Fases 1–20**: completas (pipeline base — ver historico). Sobre elas foram adicionados **Blocos A–J (#1–100)**: CFD avancado, turbulencia/multifase, transferencia de calor, acustica, otimizacao avancada, V&V e UI/DevOps.
- **Progresso**: stack completa e ampla (~66k linhas Python, 350 arquivos, 52 arquivos de teste). Aguardando deploy real + popular `training_log` com runs CFD reais.
- **Proximo marco**: `docker compose up --build` em servidor + seed `training_log` + runs CFD reais.

---

## Arquitetura Real (codigo no disco)

```
higra-pump-engine/
├── backend/
│   ├── pyproject.toml         # pacote `higra-pump-engine`, console script `hpe`
│   ├── alembic.ini            # migrations
│   └── src/hpe/
│       ├── sizing/        Meanline 1D + DB de projetos
│       │   meanline.py, specific_speed.py, velocity_triangles.py,
│       │   impeller_sizing.py, efficiency.py, cavitation.py, blade_loading.py,
│       │   convergence_solver.py, validator.py, multistage.py,
│       │   francis.py, radial_inflow_turbine.py, axial.py, axial_fan.py,
│       │   sirocco_fan.py, inducer.py, return_channel.py,
│       │   design_db.py, design_templates.py, geometry_database.py
│       ├── geometry/      Parametrica 3D (CadQuery/OCCT, fallback 2D)
│       │   parametric.py, models.py, export.py, storage.py (MinIO),
│       │   runner/ (blade, meridional), blade/, meridional/, volute/,
│       │   distributor/ (guide_vanes), draft_tube/, inverse/ (inverse design 3D)
│       ├── physics/       Off-design: euler.py, losses.py, performance.py, stability.py
│       ├── cfd/           OpenFOAM + SU2 + Ansys + malha + monitoring
│       │   pipeline.py, sweep.py, pump_curve.py, doe.py, doe_runner.py,
│       │   design_loop.py, adjoint_loop.py, domain_extent.py, geo_validator.py,
│       │   turbogrid_wrapper.py, pcf_generator.py, cfx_package.py,
│       │   openfoam/, su2/, ansys_cfx/, ansys_fluent/,
│       │   mesh/ (snappy, blockmesh, structured_blade, prism_layers, yplus, periodic, tools),
│       │   monitoring/, postprocessing/, results/
│       ├── optimization/  nsga2.py, bayesian.py, surrogate_opt.py, problem.py,
│       │   evaluator.py, optimizer.py, advanced_methods.py, enhancements.py,
│       │   doe.py, rsm.py, rrs.py
│       ├── ai/
│       │   surrogate/  v1_xgboost.py, v2_gp.py, evaluator.py, dataset.py,
│       │   │            model.py, predictor.py, eta_predictor.py
│       │   pinn/       model.py, losses.py, trainer.py
│       │   assistant/  rag.py, offline_rules.py, interpreter.py, recommender.py
│       │   anomaly/    detector.py (Isolation Forest), validators.py
│       │   training/   trainer.py, auto_train.py, experiment.py
│       ├── pipeline/      cfd_pipeline.py — geometria→malha→solver→pos
│       ├── orchestrator/  Celery (celery_app/config/tasks), status.py (Redis), versions.py
│       ├── api/           FastAPI — DOIS apps (ver abaixo) + ~42 modulos de rota
│       ├── postprocess/   openfoam_parser.py, metrics.py
│       ├── validation/    benchmark.py, benchmarks.py (SHF/ERCOFTAC/TUD)
│       ├── reports/       generator.py (PDF/HTML/MD)
│       ├── data/          bancada_etl.py, bancada_seed.py, feature_store.py, training_log.py
│       ├── core/          models.py, enums.py, config.py, db_models.py,
│       │                  database.py, persistence.py, udp.py, project_file.py
│       ├── db/            connection.py, repositories.py, migrate.py, schema.sql
│       ├── migrations/    alembic versions
│       └── infra/  io/  cli/ (advanced_commands.py)
├── frontend/          React 18 + TS + Vite + three.js + recharts (~80 componentes)
│   └── src/  App.tsx, components/, pages/, hooks/, services/, i18n/, utils/, styles/
├── docs/  dataset/  models/  mlruns/  mlflow.db  scripts/  tests/  output/  data/
├── Dockerfile  docker-compose.yml  Makefile  .env.example  conftest.py
└── CLAUDE.md  README.md
```

> **Nota**: existe `tests/` na raiz **e** `backend/tests/` (52 arquivos de teste). O empacotamento (pyproject, alembic) vive em `backend/`.

---

## API — dois apps FastAPI

| App | Modulo | Uso |
|-----|--------|-----|
| **Completo** | `hpe.api.app:app` | Aplicacao de producao. Inclui ~42 routers (auth, sizing, analysis, geometry, surrogate, inverse_design 2D/3D, optimize, io, blade, db/design_db, convergence, volute, mri, turbotype, lete, noise, batch, cfd_loop, cfd_advanced=phase_17_20, physics, infra, rrs, blade_collision, template, domain, udp, blockage, ansys, lean_sweep, version, assistant, WS pipeline/optimize). Middleware: RateLimit (120 rpm) + Multitenancy. |
| **Focado v2.0** | `hpe.api.main:app` | Sub-app enxuto da spec v2.0: `POST /sizing/run`, `POST /geometry/run`, `POST /surrogate/predict`, `GET /surrogate/similar`, `GET /health`. Bom para deploy standalone/testes. |

> Em producao, o **app completo** (`hpe.api.app:app`) e o que e servido atras do nginx.

---

## O Que NAO Existe Ainda (pendencias reais)

- [ ] `training_log` com dados CFD reais — `bancada_seed.py` popula os 460 da bancada; CFD aguarda runs reais.
- [ ] Deploy real — `docker compose up --build` no servidor + `.env` com segredos de producao.
- [ ] Modelos treinados versionados — `models/` esta no `.gitignore`; treinar surrogate v1/v2 e PINN apos seed.

---

## Bloqueios Conhecidos

- **CadQuery**: pode nao estar instalado localmente — `geometry/export.py` retorna perfis 2D com fallback gracioso. Docker: stage `backend-cad`.
- **Celery/Redis**: nao rodando localmente — orchestrator usa `_FakeTask` (sincrono). Docker: ok.
- **OpenFOAM/SU2/Ansys**: instalacoes system-level; modulos CFD tem dry-run/fallback quando ausentes.
- **Tabela bancada SIGS**: `public.hgr_lab_reg_teste` no banco `higra_sigs` (localhost:5432, **somente leitura — nunca escrever**).
- **models/ no .gitignore**: `surrogate_v1.pkl`, `surrogate_v2_gp.pkl`, `pinn_v1.pkl` ignorados.

---

## Como Executar

```bash
# API completa (producao)
PYTHONPATH=backend/src uvicorn hpe.api.app:app --port 8000 --reload

# API focada v2.0 (standalone)
PYTHONPATH=backend/src uvicorn hpe.api.main:app --port 8000 --reload

# CLI (console script `hpe` ou modulo)
PYTHONPATH=backend/src python -m hpe.cli sizing --flow 0.05 --head 30 --rpm 1750

# Testes
PYTHONPATH=backend/src pytest backend/tests/ -v
PYTHONPATH=backend/src pytest tests/ -v

# Seed training_log (460 registros bancada)
PYTHONPATH=backend/src python backend/src/hpe/data/bancada_seed.py

# Treinar PINN
PYTHONPATH=backend/src python -c "from hpe.ai.pinn.trainer import train_pinn_from_bancada; train_pinn_from_bancada(epochs=100)"

# Frontend
cd frontend && npm run dev
```

---

## Deploy — Checklist

### Pre-requisitos
```bash
cp .env.example .env
# Editar .env: substituir todos os TROQUE_* por valores reais
# Gerar SECRET_KEY: python -c "import secrets; print(secrets.token_hex(32))"
```

### Subir stack
```bash
docker compose up --build -d
docker compose ps          # aguarda ~30s para DB inicializar

# Smoke test
curl http://localhost:3000/health
curl http://localhost:3000/sizing/run \
  -H "Content-Type: application/json" \
  -d '{"Q":0.05,"H":30,"n":1750}'
```

### Seed training_log (uma vez apos deploy)
```bash
docker compose exec backend python /app/backend/src/hpe/data/bancada_seed.py
```

### Treinar modelos (apos seed)
```bash
docker compose exec backend python -c "from hpe.ai.surrogate.v1_xgboost import train; train()"
docker compose exec backend python -c "from hpe.ai.surrogate.v2_gp import train; train()"  # requer >100 registros
```

### Servicos
| Servico | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API docs | http://localhost:3000/docs |
| Flower (Celery) | http://localhost:5555 (admin:hpe2026) |
| MinIO console | http://localhost:9001 (minioadmin / senha do .env) |

---

## Historico — Fases 1–20 (pipeline base)

| Fase | O que fez | Status |
|------|-----------|--------|
| 1 | MVP: Sizing 1D, ETL bancada, Surrogate XGBoost v1, API FastAPI | DONE |
| 2 | CFD Pipeline: OpenFOAM case builder, SU2, extractor | DONE |
| 3 | Surrogate v2 GP, NSGA-II DEAP, Optuna Bayesian, surrogate-assisted | DONE |
| 4 | Voluta pipeline, training_log seed (460 registros) | DONE |
| 5 | Celery orchestrator (3 filas), Redis status, WebSocket, DesignVersion | DONE |
| 6 | PINN (PyTorch + numpy fallback), RAG assistant (Gulich KB + Claude API) | DONE |
| 7 | Frontend: PipelinePanel, tab Pipeline Completo | DONE |
| 8 | Docker: nginx proxy (WS + v2 routes), Dockerfiles, deps | DONE |
| 9 | Malha estruturada O-H (TurboGrid equiv): yplus, periodic, structured_blade | DONE |
| 10 | CadQuery Docker (backend-cad), geometry/export.py, storage.py (MinIO) | DONE |
| 11 | Multi-point CFD sweep (50-130% BEP), pump curve H-Q + eta-Q, /cfd/sweep | DONE |
| 12 | Convergencia adaptativa (ConvergenceMonitor), k-omega SST | DONE |
| 13 | Blade loading Cp PS/SS, cavitacao sigma + NPSHr + Nss | DONE |
| 14 | SU2 adjoint: runner direto+adjoint, sensitivity (dJ/dbeta2, dJ/dD2) | DONE |
| 15 | DoE LHS/Sobol/factorial, runner paralelo + retreino surrogate | DONE |
| 16 | Loop adjoint fechado, UI Cavitacao, UI Simulacao CFD | DONE |
| 17 | Paridade Ansys I: multi_domain cyclicAMI, cavitacao ZGB, prism layers, spanwise | DONE |
| 18 | Visualizacao campo CFD: VTU export, Q-criterion/streamlines, loss audit, turbo views | DONE |
| 19 | Transiente + ruido: pimpleFoam sliding mesh, FFT BPF, radial forces, gamma-Re_theta | DONE |
| 20 | Otim. avancada: mesh morphing, multi-stage, benchmarks SHF/ERCOFTAC/TUD, reports PDF | DONE |

## Historico — Blocos A–J (#1–100, sobre as fases)

| Bloco | # | Tema | Commit |
|-------|---|------|--------|
| A+B | 1–20 | Solver core + mesh avancado | 676f314 |
| C+D | 21–40 | Turbulencia + multifase avancado | 29dfca7 |
| E+F | 41–60 | Transferencia de calor + acustica | 674a510 |
| G+H | 61–80 | Otimizacao avancada + V&V | 178970c |
| I+J | 81–100 | UI avancada + DevOps | 6f66edd |

---

## Decisoes Tecnicas Tomadas

| Decisao | Escolha | Motivo |
|---------|---------|--------|
| Surrogate v1 | XGBoost | Dados limitados; RMSE 2.8% validado |
| Surrogate v2 | GP sklearn | Incerteza nativa; subsample 500pts para O(n3) |
| Otimizacao | NSGA-II + Optuna | Multi-objetivo; fallback se dep nao instalada |
| CFD | OpenFOAM + SU2 adjoint + Ansys CFX/Fluent | Industry standard; SU2 para gradiente |
| PINN | PyTorch + numpy fallback | Portabilidade sem GPU obrigatoria |
| RAG | Local KB + Claude API opcional | Offline-first; upgradeable sem mudar interface |
| Feature store | Parquet local | Migrar para S3/MinIO em producao |
| Normalizacao | StandardScaler | Compativel com GP e PINN |
| API | App completo `app.py` + sub-app `main.py` | Producao full vs deploy enxuto v2.0 |

---

## Notas de Arquitetura

- **Nunca** substituir surrogate em producao sem versionar no MLflow primeiro.
- **Sempre** registrar runs CFD no `training_log` — regra de ouro do projeto.
- Surrogate e avaliador primario no loop de otimizacao; CFD apenas para validacao final.
- Todos os modulos com fallback gracioso quando dependencias pesadas (CadQuery, Celery, Redis, PyTorch, OpenFOAM/SU2/Ansys) nao estao instaladas.
- Banco `higra_sigs` e somente leitura — nunca escrever nele.
- O nome "Pump Engine" e historico: o escopo real cobre bombas, turbinas e pump-turbines.
