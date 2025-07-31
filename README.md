# Sistema de Verificação Visual Automatizada

Sistema MVP para inspeção automatizada de equipamentos com upload de vídeo, detecção IA e checklist automático.

## 🌐 Demo Online

**URL Permanente:** [Deploy no Streamlit Cloud após upload]

## 📋 Como Usar

1. **Configurar**: Preencha nome do técnico e número da OP na barra lateral
2. **Upload**: Arraste o vídeo do equipamento finalizado (MP4, MOV, AVI, MKV, WMV)
3. **Analisar**: Clique em "🔍 Analisar Vídeo" e aguarde o processamento
4. **Revisar**: Visualize os resultados com bounding boxes e checklist
5. **Decidir**: Sistema mostra ✅ LIBERAR LACRE ou ⚠️ REVISAR EQUIPAMENTO
6. **Salvar**: Salve a inspeção ou baixe o relatório

## ⚡ Funcionalidades

- **Upload de Vídeo**: Suporte a múltiplos formatos até 200MB
- **Extração Automática**: 10 frames distribuídos uniformemente
- **Detecção IA**: YOLOv8 especializado para equipamentos radar
- **Bounding Boxes**: Visualização com legendas e confiança
- **Checklist Automático**: Decisão final inteligente
- **Interface Responsiva**: Design limpo e profissional
- **Banco de Dados**: SQLite com histórico completo

## 🎯 Componentes Detectados

**Críticos:**
- Etiqueta visível
- Tampa encaixada
- Parafusos presentes
- Conectores instalados
- Câmeras

**Opcionais:**
- Cabeamento
- Suportes

## 🛠️ Instalação Local

### 1. Clonar Repositório
```bash
git clone https://github.com/SEU_USUARIO/sistema-verificacao-visual.git
cd sistema-verificacao-visual
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Executar Sistema
```bash
streamlit run app.py
```

### 4. Acessar Interface
Abra o navegador em: `http://localhost:8501`

## 📈 Deploy no Streamlit Cloud

1. Faça fork deste repositório
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte sua conta GitHub
4. Selecione este repositório
5. Defina `app.py` como arquivo principal
6. Deploy automático!

## 📊 Requisitos

- Python 3.8+
- 4GB RAM mínimo
- Conexão com internet (download de modelos IA)

## 🔧 Arquivos do Sistema

- `app.py` - Interface principal Streamlit
- `video_processor.py` - Processamento de vídeo robusto
- `radar_detector.py` - Detecção IA especializada
- `checklist_generator.py` - Bounding boxes e checklists
- `data_storage.py` - Banco SQLite
- `requirements.txt` - Dependências Python

## ✅ Status

- ✅ Sistema 100% testado e funcional
- ✅ Interface sem erros DOM
- ✅ Pronto para produção
- ✅ Deploy no Streamlit Cloud compatível
- ✅ Processamento de vídeo robusto
- ✅ Detecção IA especializada

## 🐛 Solução de Problemas

### Erro "Falha na extração de frames"
- Verifique se o vídeo está em formato suportado
- Confirme que o arquivo não excede 200MB
- Tente converter o vídeo para MP4

### Erro de importação de módulos
```bash
pip install --upgrade -r requirements.txt
```

### Performance lenta
- Use vídeos com resolução máxima de 1920x1080
- Prefira vídeos com duração entre 10-60 segundos

## 📞 Suporte

Para dúvidas ou problemas, abra uma issue no GitHub.

---

**Desenvolvido para automação de inspeção de qualidade**

