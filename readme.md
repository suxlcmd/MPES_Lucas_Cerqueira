# 🛡️ Framework Web de Auditoria Interna: STEP-GAN & SMPC

Este repositório contém a implementação da Aplicação Web de Detecção de Anomalias Financeiras para apoio à Auditoria Interna[cite: 1]. O sistema operacionaliza o treinamento colaborativo de modelos de detecção de fraudes utilizando a arquitetura adversarial **STEP-GAN**, **Aprendizado Federado** e **Computação Segura Multipartidária (SMPC)**[cite: 1].

A solução permite que dados sensíveis sejam mantidos localmente, compartilhando apenas frações protegidas dos parâmetros para a agregação global[cite: 1].

## 🏗️ Arquitetura em Camadas

O framework foi construído seguindo uma arquitetura de 5 camadas para garantir modularidade e segurança[cite: 1]:
1. **Camada de Apresentação:** Interface interativa construída com Streamlit[cite: 1].
2. **Camada de Aplicação e Orquestração:** Gerenciamento de pipelines de dados e coordenação do aprendizado federado[cite: 1].
3. **Camada de Aprendizado de Máquina:** Implementação em PyTorch da arquitetura STEP-GAN (Gerador e Discriminador)[cite: 1].
4. **Camada Criptográfica e de Privacidade:** Compartilhamento Aditivo de Segredos (SMPC) em aritmética modular[cite: 1].
5. **Camada de Dados e Nuvem:** Persistência e versionamento de artefatos no Azure Blob Storage[cite: 1].

## 📂 Estrutura de Diretórios

```text
/
├── .env                        # Variáveis de ambiente (NÃO COMMITAR)
├── .gitignore                  # Arquivos ignorados pelo Git
├── app.py                      # Ponto de entrada da aplicação Streamlit
├── views/                      # Camada 1: Interfaces de usuário
│   ├── dashboard.py
│   ├── training.py
│   └── azure_manager.py
├── orchestration/              # Camada 2: Processamento e controle federado
│   ├── data_preprocessor.py
│   └── federated_coordinator.py
├── ml_engine/                  # Camada 3: Modelos PyTorch
│   ├── generator.py
│   ├── discriminator.py
│   └── step_gan_trainer.py
├── privacy/                    # Camada 4: Criptografia e agregação
│   ├── secret_sharing.py
│   └── smpc_aggregator.py
└── cloud/                      # Camada 5: Conectividade
    └── azure_blob_client.py


⚙️ Configuração do Ambiente
1. Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado em sua máquina.

2. Instalação das Dependências
Instale as bibliotecas necessárias executando o comando abaixo no terminal:

pip install streamlit pandas numpy scikit-learn torch azure-storage-blob python-dotenv

3. Configuração de Credenciais (Azure)
Para proteger suas chaves de acesso à nuvem, o projeto utiliza variáveis de ambiente.
Crie um arquivo chamado .env na raiz do projeto e adicione sua string de conexão do Azure Blob Storage:

AZURE_CONNECTION_STRING="SuaStringDeConexaoAqui"

🚀 Como Executar
Com o ambiente configurado e as dependências instaladas, inicie a aplicação web executando:

streamlit run app.py