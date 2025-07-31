#!/usr/bin/env python3
"""
Gerador de Checklist Automático - Sistema de Verificação Visual
Gera checklists preenchidos e relatórios visuais com bounding boxes
"""

import cv2
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import json
from PIL import Image, ImageDraw, ImageFont
import os

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChecklistGenerator:
    """
    Gerador de checklist automático com visualizações
    """
    
    def __init__(self):
        """Inicializa o gerador de checklist"""
        self.colors = {
            'detected': (0, 255, 0),      # Verde para detectado
            'missing': (255, 0, 0),       # Vermelho para não detectado
            'warning': (255, 165, 0),     # Laranja para aviso
            'text': (255, 255, 255),      # Branco para texto
            'background': (0, 0, 0)       # Preto para fundo do texto
        }
        
        self.component_positions = {
            'etiqueta_visivel': (0.1, 0.1, 0.4, 0.3),      # top-left
            'tampa_encaixada': (0.0, 0.0, 1.0, 1.0),       # full frame
            'parafusos_presentes': (0.1, 0.7, 0.3, 0.9),   # bottom-left
            'conectores_instalados': (0.7, 0.7, 0.9, 0.9), # bottom-right
            'cabeamento': (0.3, 0.4, 0.7, 0.8),            # center
            'cameras': (0.4, 0.1, 0.6, 0.4),               # top-center
            'suportes': (0.0, 0.3, 1.0, 0.7)               # middle strip
        }
        
        logger.info("Gerador de checklist inicializado")
    
    def draw_bounding_boxes(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """
        Desenha bounding boxes com legendas e confiança
        
        Args:
            frame: Frame original
            analysis_result: Resultado da análise
            
        Returns:
            np.ndarray: Frame com bounding boxes
        """
        # Criar cópia do frame
        annotated_frame = frame.copy()
        
        try:
            components = analysis_result.get('components', {})
            
            for component_name, component_data in components.items():
                detected = component_data.get('detected', False)
                confidence = component_data.get('confidence', 0.0)
                
                # Obter posição do componente
                if component_name in self.component_positions:
                    x1_rel, y1_rel, x2_rel, y2_rel = self.component_positions[component_name]
                    
                    # Converter para coordenadas absolutas
                    h, w = frame.shape[:2]
                    x1 = int(x1_rel * w)
                    y1 = int(y1_rel * h)
                    x2 = int(x2_rel * w)
                    y2 = int(y2_rel * h)
                    
                    # Escolher cor baseada no status
                    if detected:
                        color = self.colors['detected']
                        status = "OK"
                    else:
                        color = self.colors['missing']
                        status = "MISSING"
                    
                    # Desenhar bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Preparar texto
                    component_display = component_name.replace('_', ' ').title()
                    label = f"{component_display}: {status}"
                    confidence_text = f"Conf: {confidence:.1%}"
                    
                    # Calcular posição do texto
                    font_scale = 0.6
                    thickness = 2
                    
                    # Tamanho do texto principal
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    
                    # Tamanho do texto de confiança
                    (conf_width, conf_height), _ = cv2.getTextSize(
                        confidence_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness
                    )
                    
                    # Posição do fundo do texto
                    text_x = x1
                    text_y = y1 - 10
                    
                    # Ajustar se o texto sair da imagem
                    if text_y < text_height + 5:
                        text_y = y2 + text_height + 5
                    
                    # Desenhar fundo do texto
                    bg_x1 = text_x - 5
                    bg_y1 = text_y - text_height - 5
                    bg_x2 = text_x + max(text_width, conf_width) + 10
                    bg_y2 = text_y + conf_height + 10
                    
                    cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                                self.colors['background'], -1)
                    
                    # Desenhar texto principal
                    cv2.putText(annotated_frame, label, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                              self.colors['text'], thickness)
                    
                    # Desenhar texto de confiança
                    cv2.putText(annotated_frame, confidence_text, 
                              (text_x, text_y + conf_height + 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, 
                              self.colors['text'], thickness)
            
        except Exception as e:
            logger.error(f"Erro ao desenhar bounding boxes: {e}")
        
        return annotated_frame
    
    def generate_frame_checklist(self, frame_number: int, analysis_result: Dict) -> Dict:
        """
        Gera checklist para um frame específico
        
        Args:
            frame_number: Número do frame
            analysis_result: Resultado da análise
            
        Returns:
            Dict: Checklist do frame
        """
        checklist = {
            'frame_number': frame_number,
            'timestamp': analysis_result.get('timestamp'),
            'overall_score': analysis_result.get('overall_score', 0.0),
            'status': analysis_result.get('status', 'UNKNOWN'),
            'items': []
        }
        
        try:
            components = analysis_result.get('components', {})
            
            for component_name, component_data in components.items():
                item = {
                    'component': component_name.replace('_', ' ').title(),
                    'detected': component_data.get('detected', False),
                    'confidence': component_data.get('confidence', 0.0),
                    'critical': component_data.get('critical', False),
                    'details': component_data.get('details', ''),
                    'status_icon': '✓' if component_data.get('detected') else '✗',
                    'status_text': 'DETECTADO' if component_data.get('detected') else 'NÃO DETECTADO'
                }
                
                checklist['items'].append(item)
            
            # Ordenar por criticidade e depois por nome
            checklist['items'].sort(key=lambda x: (not x['critical'], x['component']))
            
        except Exception as e:
            logger.error(f"Erro ao gerar checklist do frame: {e}")
        
        return checklist
    
    def generate_consolidated_checklist(self, all_analyses: List[Dict], 
                                      video_info: Dict, operator_info: Dict) -> Dict:
        """
        Gera checklist consolidado de todos os frames
        
        Args:
            all_analyses: Lista de análises de todos os frames
            video_info: Informações do vídeo
            operator_info: Informações do operador
            
        Returns:
            Dict: Checklist consolidado
        """
        consolidated = {
            'inspection_info': {
                'operator_name': operator_info.get('operator_name', ''),
                'op_number': operator_info.get('op_number', ''),
                'video_filename': video_info.get('filename', ''),
                'inspection_date': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                'total_frames': len(all_analyses),
                'video_duration': f"{video_info.get('duration', 0):.1f}s"
            },
            'summary': {
                'overall_score': 0.0,
                'final_decision': 'UNKNOWN',
                'critical_components_status': 'UNKNOWN',
                'approved_frames': 0,
                'rejected_frames': 0,
                'review_frames': 0
            },
            'components_analysis': {},
            'frame_by_frame': []
        }
        
        try:
            # Calcular estatísticas gerais
            total_frames = len(all_analyses)
            approved_frames = sum(1 for a in all_analyses if a.get('status') == 'APROVADO')
            rejected_frames = sum(1 for a in all_analyses if a.get('status') == 'REPROVADO')
            review_frames = total_frames - approved_frames - rejected_frames
            
            avg_score = np.mean([a.get('overall_score', 0) for a in all_analyses])
            
            consolidated['summary'].update({
                'overall_score': avg_score,
                'approved_frames': approved_frames,
                'rejected_frames': rejected_frames,
                'review_frames': review_frames
            })
            
            # Analisar cada componente
            component_names = set()
            for analysis in all_analyses:
                component_names.update(analysis.get('components', {}).keys())
            
            for component_name in component_names:
                component_analysis = {
                    'component_name': component_name.replace('_', ' ').title(),
                    'detected_in_frames': 0,
                    'total_frames': total_frames,
                    'detection_rate': 0.0,
                    'average_confidence': 0.0,
                    'critical': False,
                    'final_status': 'NOT_DETECTED',
                    'best_detection': None,
                    'frame_detections': []
                }
                
                detections = []
                confidences = []
                
                for i, analysis in enumerate(all_analyses):
                    components = analysis.get('components', {})
                    if component_name in components:
                        comp_data = components[component_name]
                        detected = comp_data.get('detected', False)
                        confidence = comp_data.get('confidence', 0.0)
                        
                        component_analysis['critical'] = comp_data.get('critical', False)
                        
                        frame_detection = {
                            'frame_number': i + 1,
                            'detected': detected,
                            'confidence': confidence,
                            'details': comp_data.get('details', '')
                        }
                        
                        component_analysis['frame_detections'].append(frame_detection)
                        
                        if detected:
                            detections.append(True)
                            confidences.append(confidence)
                            
                            # Atualizar melhor detecção
                            if (component_analysis['best_detection'] is None or 
                                confidence > component_analysis['best_detection']['confidence']):
                                component_analysis['best_detection'] = frame_detection
                        else:
                            detections.append(False)
                            confidences.append(0.0)
                
                # Calcular estatísticas do componente
                component_analysis['detected_in_frames'] = sum(detections)
                component_analysis['detection_rate'] = sum(detections) / len(detections) if detections else 0
                component_analysis['average_confidence'] = np.mean(confidences) if confidences else 0
                
                # Determinar status final do componente
                if component_analysis['detected_in_frames'] > 0:
                    if component_analysis['detection_rate'] >= 0.5:
                        component_analysis['final_status'] = 'DETECTED'
                    else:
                        component_analysis['final_status'] = 'PARTIALLY_DETECTED'
                else:
                    component_analysis['final_status'] = 'NOT_DETECTED'
                
                consolidated['components_analysis'][component_name] = component_analysis
            
            # Determinar decisão final
            critical_components = [comp for comp in consolidated['components_analysis'].values() 
                                 if comp['critical']]
            
            all_critical_detected = all(comp['final_status'] == 'DETECTED' 
                                      for comp in critical_components)
            
            if all_critical_detected and avg_score >= 0.7:
                consolidated['summary']['final_decision'] = 'LIBERAR_LACRE'
                consolidated['summary']['critical_components_status'] = 'TODOS_OK'
            elif avg_score >= 0.5:
                consolidated['summary']['final_decision'] = 'REVISAR_EQUIPAMENTO'
                consolidated['summary']['critical_components_status'] = 'PARCIAL'
            else:
                consolidated['summary']['final_decision'] = 'REPROVAR_EQUIPAMENTO'
                consolidated['summary']['critical_components_status'] = 'FALHA'
            
            # Gerar checklist frame por frame
            for i, analysis in enumerate(all_analyses):
                frame_checklist = self.generate_frame_checklist(i + 1, analysis)
                consolidated['frame_by_frame'].append(frame_checklist)
            
        except Exception as e:
            logger.error(f"Erro ao gerar checklist consolidado: {e}")
        
        return consolidated
    
    def format_checklist_for_display(self, consolidated_checklist: Dict) -> str:
        """
        Formata checklist para exibição em texto
        
        Args:
            consolidated_checklist: Checklist consolidado
            
        Returns:
            str: Checklist formatado
        """
        try:
            info = consolidated_checklist['inspection_info']
            summary = consolidated_checklist['summary']
            components = consolidated_checklist['components_analysis']
            
            # Cabeçalho
            report = f"""
CHECKLIST DE VERIFICAÇÃO VISUAL AUTOMATIZADA
============================================

INFORMAÇÕES DA INSPEÇÃO:
- Operador: {info['operator_name']}
- OP: {info['op_number']}
- Data/Hora: {info['inspection_date']}
- Arquivo: {info['video_filename']}
- Duração: {info['video_duration']}
- Frames Analisados: {info['total_frames']}

RESUMO EXECUTIVO:
- Score Geral: {summary['overall_score']:.1%}
- Decisão Final: {summary['final_decision'].replace('_', ' ')}
- Status Componentes Críticos: {summary['critical_components_status'].replace('_', ' ')}
- Frames Aprovados: {summary['approved_frames']}/{info['total_frames']}
- Frames Reprovados: {summary['rejected_frames']}/{info['total_frames']}

ANÁLISE POR COMPONENTE:
"""
            
            # Componentes críticos primeiro
            critical_components = [comp for comp in components.values() if comp['critical']]
            optional_components = [comp for comp in components.values() if not comp['critical']]
            
            report += "\nCOMPONENTES CRÍTICOS:\n"
            report += "-" * 50 + "\n"
            
            for comp in critical_components:
                status_icon = "✓" if comp['final_status'] == 'DETECTED' else "✗"
                report += f"{status_icon} {comp['component_name']}\n"
                report += f"   Status: {comp['final_status'].replace('_', ' ')}\n"
                report += f"   Detectado em: {comp['detected_in_frames']}/{comp['total_frames']} frames\n"
                report += f"   Taxa de Detecção: {comp['detection_rate']:.1%}\n"
                report += f"   Confiança Média: {comp['average_confidence']:.1%}\n"
                
                if comp['best_detection']:
                    best = comp['best_detection']
                    report += f"   Melhor Detecção: Frame {best['frame_number']} ({best['confidence']:.1%})\n"
                
                report += "\n"
            
            report += "\nCOMPONENTES OPCIONAIS:\n"
            report += "-" * 50 + "\n"
            
            for comp in optional_components:
                status_icon = "✓" if comp['final_status'] == 'DETECTED' else "○"
                report += f"{status_icon} {comp['component_name']}\n"
                report += f"   Status: {comp['final_status'].replace('_', ' ')}\n"
                report += f"   Detectado em: {comp['detected_in_frames']}/{comp['total_frames']} frames\n"
                report += f"   Taxa de Detecção: {comp['detection_rate']:.1%}\n\n"
            
            # Decisão final
            report += "\nDECISÃO FINAL:\n"
            report += "=" * 50 + "\n"
            
            decision = summary['final_decision']
            if decision == 'LIBERAR_LACRE':
                report += "✓ EQUIPAMENTO APROVADO - LIBERAR PARA LACRE\n"
                report += "  Todos os componentes críticos foram detectados.\n"
            elif decision == 'REVISAR_EQUIPAMENTO':
                report += "⚠ EQUIPAMENTO REQUER REVISÃO\n"
                report += "  Alguns componentes críticos não foram detectados adequadamente.\n"
            else:
                report += "✗ EQUIPAMENTO REPROVADO\n"
                report += "  Falhas críticas detectadas. Revisar montagem.\n"
            
            report += f"\nScore Geral: {summary['overall_score']:.1%}\n"
            report += f"Data: {info['inspection_date']}\n"
            
        except Exception as e:
            logger.error(f"Erro ao formatar checklist: {e}")
            report = f"Erro ao gerar relatório: {e}"
        
        return report
    
    def save_annotated_frames(self, frames: List[np.ndarray], analyses: List[Dict], 
                            output_dir: str) -> List[str]:
        """
        Salva frames com anotações
        
        Args:
            frames: Lista de frames
            analyses: Lista de análises
            output_dir: Diretório de saída
            
        Returns:
            List[str]: Lista de caminhos dos arquivos salvos
        """
        saved_paths = []
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, (frame, analysis) in enumerate(zip(frames, analyses)):
                # Desenhar bounding boxes
                annotated_frame = self.draw_bounding_boxes(frame, analysis)
                
                # Salvar frame anotado
                filename = f"frame_annotated_{timestamp}_{i+1:02d}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Converter RGB para BGR para salvar com OpenCV
                frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                success = cv2.imwrite(filepath, frame_bgr)
                
                if success:
                    saved_paths.append(filepath)
                    logger.info(f"Frame anotado {i+1} salvo: {filepath}")
                else:
                    logger.error(f"Falha ao salvar frame anotado {i+1}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar frames anotados: {e}")
        
        return saved_paths


def test_checklist_generator():
    """Função de teste do gerador de checklist"""
    generator = ChecklistGenerator()
    
    print("=== TESTE DO GERADOR DE CHECKLIST ===")
    
    # Criar dados de teste
    mock_analysis = {
        'timestamp': datetime.now().isoformat(),
        'overall_score': 0.85,
        'status': 'APROVADO',
        'components': {
            'etiqueta_visivel': {
                'detected': True,
                'confidence': 0.9,
                'critical': True,
                'details': 'Etiqueta detectada'
            },
            'tampa_encaixada': {
                'detected': True,
                'confidence': 0.8,
                'critical': True,
                'details': 'Tampa bem encaixada'
            },
            'parafusos_presentes': {
                'detected': False,
                'confidence': 0.3,
                'critical': True,
                'details': 'Parafusos não detectados'
            }
        }
    }
    
    # Testar checklist de frame
    frame_checklist = generator.generate_frame_checklist(1, mock_analysis)
    print(f"Checklist do frame gerado: {len(frame_checklist['items'])} itens")
    
    # Testar checklist consolidado
    video_info = {'filename': 'test.mp4', 'duration': 30.0}
    operator_info = {'operator_name': 'João Silva', 'op_number': 'OP-001'}
    
    consolidated = generator.generate_consolidated_checklist(
        [mock_analysis], video_info, operator_info
    )
    
    print(f"Checklist consolidado gerado")
    print(f"Decisão final: {consolidated['summary']['final_decision']}")
    
    # Testar formatação
    formatted = generator.format_checklist_for_display(consolidated)
    print(f"Relatório formatado: {len(formatted)} caracteres")
    
    print("Gerador de checklist testado com sucesso!")


if __name__ == "__main__":
    test_checklist_generator()

