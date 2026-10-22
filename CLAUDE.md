# Contexto e Arquitetura do Projeto

## Papel de Cada Agente neste Repositório (Tiered Routing)
- **Antigravity CLI (`agy` / Gemini 4 Argon & 3.8 Flash):** Responsável por indexação de contexto amplo, leitura de múltiplos arquivos/logs, exploração arquitetural e criação/atualização do arquivo `PLAN.md`.
- **Claude Code (`claude`):** Responsável por execução cirúrgica, refatorações complexas, implementação guiada pelo `PLAN.md`, resolução de bugs profundos e execução de testes.
- **Local (`qwen-code` via Ollama):** Tarefas rápidas inline, docstrings, formatação e scripts auxiliares.

## Regras Gerais de Código
- Antes de grandes mudanças estruturais, verifique se existe um `PLAN.md` na raiz do projeto.
- Mantenha alterações focadas no escopo solicitado sem refatorar código não relacionado.
- Sempre valide alterações rodando a suíte de testes ou linter correspondente antes de finalizar.

## Comandos do Projeto (Preencher por repositório)
- **Ambiente / Build:** `definir`
- **Testes:** `pytest` (ou equivalente do projeto)
- **Linter / Typecheck:** `definir`
