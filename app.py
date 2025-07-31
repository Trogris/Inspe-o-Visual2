
#!/usr/bin/env python3
"""
Sistema de Verificação Visual Automatizada
Interface Streamlit - Versão Funcional Corrigida
"""

# [Importações, configuração e demais funções mantidas como estavam...]

# Correção aplicada na função render_analysis_results()
def render_analysis_results():
    """Renderiza resultados da análise"""
    if not st.session_state.consolidated_checklist:
        return

    checklist = st.session_state.consolidated_checklist

    st.subheader("Resultados da Análise")

    # Decisão final
    decision = checklist['summary']['final_decision']
    score = checklist['summary']['overall_score']

    if decision == "LIBERAR_LACRE":
        st.markdown(f"""
        <div class="status-approved">
            ✅ LIBERAR LACRE<br>
            Equipamento aprovado com score: {score:.1f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-review">
            ⚠️ REVISAR EQUIPAMENTO<br>
            Equipamento precisa de revisão. Score: {score:.1f}%
        </div>
        """, unsafe_allow_html=True)

    # Resumo dos componentes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Componentes Críticos")
        for component, data in checklist['components_analysis'].items():
            if data.get('critical', False):
                status = "✅" if data.get('detected', False) else "❌"
                confidence = data.get('confidence', 0)
                st.write(f"{status} **{component.replace('_', ' ').title()}**: {confidence:.1f}%")

    with col2:
        st.subheader("Componentes Opcionais")
        for component, data in checklist['components_analysis'].items():
            if not data.get('critical', True):
                status = "✅" if data.get('detected', False) else "❌"
                confidence = data.get('confidence', 0)
                st.write(f"{status} **{component.replace('_', ' ').title()}**: {confidence:.1f}%")

    # [Demais seções da função mantidas como estavam...]
