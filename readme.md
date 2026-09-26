# 🛡️ Framework Web de Auditoria Interna: STEP-GAN & SMPC

Este repositório contém a implementação da Aplicação Web de Detecção de Anomalias Financeiras para apoio à Auditoria Interna. O sistema operacionaliza o treinamento colaborativo de modelos de detecção de fraudes utilizando a arquitetura adversarial **STEP-GAN**, **Aprendizado Federado** e **Computação Segura Multipartidária (SMPC)**.

A solução permite que dados sensíveis sejam mantidos localmente, compartilhando apenas frações protegidas dos parâmetros para a agregação global.

## 🏗️ Arquitetura em Camadas

1. **Camada de Apresentação:** interface em Streamlit (`app.py`, `views/`).
2. **Camada de Aplicação e Orquestração:** pré-processamento, coordenação federada, treino, inferência, checkpoint e trilha de auditoria (`orchestration/`).
3. **Camada de Aprendizado de Máquina:** STEP-GAN em PyTorch (`ml_engine/`).
4. **Camada Criptográfica e de Privacidade:** compartilhamento aditivo de segredos em aritmética modular (`privacy/`).
5. **Camada de Dados e Nuvem:** persistência e versionamento de artefatos no Azure Blob Storage (`cloud/`).

## 🧠 Modelo e protocolo (alinhados ao Teste 4 v5 da PoC)

- **STEP-GAN:** 3 geradores condicionados por limiares `G_k(z, τ)`, um por degrau da saída do Discriminador, e **Discriminador graduado** (normal → 0,9; fraude → 0,1; amostra do degrau τ → τ).
- **Aprendizado Federado:** a cada rodada todos os clientes partem do mesmo modelo global; Discriminador e geradores são agregados por FedAvg ponderado.
- **SMPC:** campo Z_p com p = 2^61−1, ponto fixo com arredondamento (2^-24), checagem de estouro antes do compartilhamento e máscaras geradas por CSPRNG. Cada nó soma só as partes que recebe; o servidor vê apenas o agregado.
- **Sem vazamento de dados:** split 60/20/20; scaler ajustado só no treino; limiar calibrado só na validação e salvo com o modelo; métricas reportadas no teste reservado.

## 🔎 Funcionalidades para a auditoria interna

- **Checkpoint completo** (`model-checkpoints/<modelo>.pt`): pesos, pré-processador, limiar, métricas e metadados. É lido com `torch.load(weights_only=True)`, então um arquivo adulterado não executa código ao ser carregado.
- **Análise de transações** com escore de risco (0–100) independente do lote, níveis de risco (Crítico, Alto, Médio, Baixo) e **fatores mais atípicos** de cada transação.
- **Fila de triagem**: o auditor registra o status (Pendente, Em análise, Fraude confirmada, Falso positivo) e observações; o relatório é exportado em CSV ou salvo em `audit-artifacts/`.
- **Trilha de auditoria**: treinamentos, análises, uploads e relatórios geram eventos JSON (data/hora UTC, usuário, SHA-256 do arquivo, parâmetros e resultados) em `audit-artifacts/`, sem sobrescrita.
- **Dashboard** com desempenho do modelo, convergência federada, custo e exatidão do SMPC, condicionamento do STEP-GAN e status da triagem.

## 📂 Estrutura de Diretórios

```text
/
├── app.py                      # Ponto de entrada Streamlit (navegação e controle de acesso)
├── theme.py                    # Tema visual e componentes HTML (com escape de texto)
├── views/                      # Camada 1
│   ├── training.py             #   treino, carga de modelo, análise e triagem
│   ├── dashboard.py            #   indicadores, desempenho, SMPC e trilha de auditoria
│   ├── azure_manager.py        #   gestão dos artefatos no Azure
│   └── common.py               #   componentes compartilhados
├── orchestration/              # Camada 2
│   ├── data_preprocessor.py    #   esquemas Credit Card / PaySim / genérico
│   ├── federated_coordinator.py#   FL com agregação SMPC
│   ├── training_service.py     #   pipeline de treino e calibração do limiar
│   ├── inference_service.py    #   escore, níveis de risco e fatores atípicos
│   ├── model_bundle.py         #   checkpoint seguro do modelo
│   └── audit_log.py            #   eventos da trilha de auditoria
├── ml_engine/                  # Camada 3: gerador, discriminador e treino STEP-GAN
├── privacy/                    # Camada 4: secret sharing e agregador SMPC
├── cloud/                      # Camada 5: Azure Blob Storage
├── tests/                      # Testes automatizados (pytest)
└── PoC/                        # Notebooks da prova de conceito
```

## ⚙️ Configuração do Ambiente

1. Python 3.10+.
2. Dependências: `pip install -r requirements.txt` (o PyTorch é instalado na versão só para CPU).
3. Variáveis de ambiente (no App Service, em *Configurações > Variáveis de ambiente*; localmente, no arquivo `.env`, que não deve ser versionado):

| Variável | Obrigatória | Descrição |
|---|---|---|
| `AZURE_CONNECTION_STRING` | não* | Conexão com o Azure Blob Storage. Sem ela, a aplicação funciona com upload local. |
| `AZURE_CONTAINER_NAME` | não | Container dos artefatos (padrão: `mpes`). |
| `MPES_ACCESS_PASSWORD` | não | Se definida, exige senha para abrir a aplicação. |
| `MPES_DEFAULT_USER` | não | Nome registrado na trilha quando não há autenticação do App Service. |

\* Necessária para salvar checkpoints, relatórios e a trilha de auditoria na nuvem.

**Segurança recomendada em produção:** habilitar a autenticação do App Service (Easy Auth / Microsoft Entra ID). A aplicação lê o usuário autenticado do cabeçalho `X-MS-CLIENT-PRINCIPAL-NAME` e o registra na trilha de auditoria.

## 🚀 Como Executar

```bash
streamlit run app.py
```

## ✅ Testes

```bash
pip install -r requirements-dev.txt
pytest -q
```

Os testes cobrem o SMPC (negativos, estouro, partes uniformes), o pré-processamento, o pipeline federado, o checkpoint seguro, a inferência e a renderização das telas (Streamlit AppTest).
