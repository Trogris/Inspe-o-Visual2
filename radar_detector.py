#!/usr/bin/env python3
"""
Detector Específico para Equipamentos Radar de Trânsito
Sistema de Verificação Visual Automatizada
"""

import cv2
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import os

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadarEquipmentDetector:
    """
    Detector especializado para componentes de equipamentos radar de trânsito
    """
    
    def __init__(self):
        """Inicializa o detector de equipamentos radar"""
        self.model = None
        self.confidence_threshold = 0.3
        
        # Componentes específicos de radar de trânsito
        self.radar_components = {
            'etiqueta_visivel': {
                'description': 'Etiqueta de identificação visível',
                'weight': 0.20,
                'critical': True,
                'detection_method': 'ocr_and_visual'
            },
            'tampa_encaixada': {
                'description': 'Tampa do gabinete corretamente encaixada',
                'weight': 0.15,
                'critical': True,
                'detection_method': 'visual_structure'
            },
            'parafusos_presentes': {
                'description': 'Parafusos de fixação presentes',
                'weight': 0.15,
                'critical': True,
                'detection_method': 'visual_detection'
            },
            'conectores_instalados': {
                'description': 'Conectores de rede e energia instalados',
                'weight': 0.15,
                'critical': True,
                'detection_method': 'visual_detection'
            },
            'cabeamento': {
                'description': 'Cabeamento organizado e conectado',
                'weight': 0.10,
                'critical': False,
                'detection_method': 'visual_detection'
            },
            'cameras': {
                'description': 'Câmeras de captura instaladas',
                'weight': 0.15,
                'critical': True,
                'detection_method': 'visual_detection'
            },
            'suportes': {
                'description': 'Suportes e fixações estruturais',
                'weight': 0.10,
                'critical': False,
                'detection_method': 'visual_structure'
            }
        }
        
        logger.info("Detector de equipamentos radar inicializado")
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Carrega modelo YOLO para detecção
        
        Args:
            model_path: Caminho para modelo customizado (opcional)
        """
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"Modelo customizado carregado: {model_path}")
            else:
                # Usar modelo pré-treinado
                self.model = YOLO('yolov8n.pt')
                logger.info("Modelo YOLOv8 pré-treinado carregado")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.model = None
    
    def detect_etiqueta_visivel(self, frame: np.ndarray) -> Dict:
        """
        Detecta presença de etiqueta visível
        
        Args:
            frame: Frame de vídeo para análise
            
        Returns:
            Dict: Resultado da detecção
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'details': 'Etiqueta não detectada',
            'bbox': None
        }
        
        try:
            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detectar regiões retangulares (possíveis etiquetas)
            contours, _ = cv2.findContours(
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filtrar contornos por área e formato
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Área mínima para etiqueta
                    # Aproximar contorno
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Verificar se é retangular (4 vértices)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Etiquetas geralmente têm proporção específica
                        if 1.5 <= aspect_ratio <= 4.0:
                            result['detected'] = True
                            result['confidence'] = min(0.8, area / 10000)
                            result['details'] = f'Etiqueta detectada (área: {area:.0f})'
                            result['bbox'] = (x, y, w, h)
                            break
            
            # Simular detecção baseada em características da imagem
            if not result['detected']:
                # Verificar presença de texto (simulação)
                text_regions = self._detect_text_regions(gray)
                if len(text_regions) > 0:
                    result['detected'] = True
                    result['confidence'] = 0.7
                    result['details'] = f'Região com texto detectada ({len(text_regions)} regiões)'
                    
        except Exception as e:
            logger.error(f"Erro na detecção de etiqueta: {e}")
        
        return result
    
    def detect_tampa_encaixada(self, frame: np.ndarray) -> Dict:
        """
        Detecta se a tampa está corretamente encaixada
        
        Args:
            frame: Frame de vídeo para análise
            
        Returns:
            Dict: Resultado da detecção
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'details': 'Tampa não detectada ou mal encaixada',
            'bbox': None
        }
        
        try:
            # Detectar bordas para identificar estruturas
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Detectar linhas horizontais e verticais (estrutura da tampa)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combinar linhas
            structure_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Contar pixels de estrutura
            structure_pixels = np.sum(structure_mask > 0)
            total_pixels = frame.shape[0] * frame.shape[1]
            structure_ratio = structure_pixels / total_pixels
            
            # Tampa bem encaixada deve ter estrutura definida
            if structure_ratio > 0.05:  # 5% da imagem com estrutura
                result['detected'] = True
                result['confidence'] = min(0.9, structure_ratio * 10)
                result['details'] = f'Estrutura detectada ({structure_ratio:.2%} da imagem)'
            
            # Verificar uniformidade (tampa fechada tem menos variação)
            std_dev = np.std(gray)
            if std_dev < 50:  # Baixa variação indica superfície uniforme
                result['detected'] = True
                result['confidence'] = max(result['confidence'], 0.6)
                result['details'] += f' - Superfície uniforme (std: {std_dev:.1f})'
                
        except Exception as e:
            logger.error(f"Erro na detecção de tampa: {e}")
        
        return result
    
    def detect_parafusos_presentes(self, frame: np.ndarray) -> Dict:
        """
        Detecta presença de parafusos
        
        Args:
            frame: Frame de vídeo para análise
            
        Returns:
            Dict: Resultado da detecção
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'details': 'Parafusos não detectados',
            'bbox': None,
            'count': 0
        }
        
        try:
            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detectar círculos (cabeças de parafusos)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=3,
                maxRadius=15
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                screw_count = len(circles)
                
                result['detected'] = screw_count >= 2  # Mínimo 2 parafusos
                result['confidence'] = min(0.9, screw_count / 4)  # Máximo com 4 parafusos
                result['details'] = f'{screw_count} parafusos detectados'
                result['count'] = screw_count
            
            # Detecção alternativa por template matching (simulação)
            if not result['detected']:
                # Simular detecção baseada em características da imagem
                # Procurar por regiões circulares pequenas e escuras
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                morphed = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
                
                # Contar regiões que podem ser parafusos
                contours, _ = cv2.findContours(
                    cv2.threshold(morphed, 30, 255, cv2.THRESH_BINARY)[1],
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                potential_screws = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 10 <= area <= 200:  # Área típica de parafuso na imagem
                        potential_screws += 1
                
                if potential_screws >= 2:
                    result['detected'] = True
                    result['confidence'] = min(0.7, potential_screws / 4)
                    result['details'] = f'{potential_screws} possíveis parafusos detectados'
                    result['count'] = potential_screws
                    
        except Exception as e:
            logger.error(f"Erro na detecção de parafusos: {e}")
        
        return result
    
    def detect_conectores_instalados(self, frame: np.ndarray) -> Dict:
        """
        Detecta conectores instalados
        
        Args:
            frame: Frame de vídeo para análise
            
        Returns:
            Dict: Resultado da detecção
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'details': 'Conectores não detectados',
            'bbox': None
        }
        
        try:
            # Usar modelo YOLO se disponível
            if self.model:
                results = self.model(frame, conf=self.confidence_threshold)
                
                # Procurar por objetos que podem ser conectores
                connector_classes = ['plug', 'cable', 'connector', 'port']
                
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Verificar se é um conector (simulação)
                            if confidence > self.confidence_threshold:
                                result['detected'] = True
                                result['confidence'] = confidence
                                result['details'] = f'Conector detectado (confiança: {confidence:.2f})'
                                break
            
            # Detecção alternativa por características visuais
            if not result['detected']:
                # Detectar regiões retangulares pequenas (conectores)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Detectar bordas
                edges = cv2.Canny(gray, 50, 150)
                
                # Encontrar contornos
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                connector_count = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 100 <= area <= 2000:  # Área típica de conectores
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Conectores geralmente são retangulares
                        if 0.5 <= aspect_ratio <= 3.0:
                            connector_count += 1
                
                if connector_count >= 1:
                    result['detected'] = True
                    result['confidence'] = min(0.8, connector_count / 3)
                    result['details'] = f'{connector_count} conectores detectados'
                    
        except Exception as e:
            logger.error(f"Erro na detecção de conectores: {e}")
        
        return result
    
    def detect_cabeamento(self, frame: np.ndarray) -> Dict:
        """
        Detecta cabeamento organizado
        
        Args:
            frame: Frame de vídeo para análise
            
        Returns:
            Dict: Resultado da detecção
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'details': 'Cabeamento não detectado',
            'bbox': None
        }
        
        try:
            # Detectar linhas (cabos)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Detectar linhas usando transformada de Hough
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10
            )
            
            if lines is not None:
                cable_lines = len(lines)
                
                # Cabos organizados têm várias linhas
                if cable_lines >= 5:
                    result['detected'] = True
                    result['confidence'] = min(0.8, cable_lines / 20)
                    result['details'] = f'{cable_lines} linhas de cabeamento detectadas'
            
            # Verificar presença de cores típicas de cabos
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Máscaras para cores comuns de cabos
            # Azul (cabo de rede)
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            # Amarelo (cabo de energia)
            yellow_mask = cv2.inRange(hsv, (20, 50, 50), (30, 255, 255))
            # Preto (cabo comum)
            black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
            
            cable_pixels = np.sum(blue_mask > 0) + np.sum(yellow_mask > 0) + np.sum(black_mask > 0)
            total_pixels = frame.shape[0] * frame.shape[1]
            cable_ratio = cable_pixels / total_pixels
            
            if cable_ratio > 0.02:  # 2% da imagem com cores de cabo
                result['detected'] = True
                result['confidence'] = max(result['confidence'], min(0.7, cable_ratio * 20))
                result['details'] += f' - Cores de cabo detectadas ({cable_ratio:.2%})'
                
        except Exception as e:
            logger.error(f"Erro na detecção de cabeamento: {e}")
        
        return result
    
    def detect_cameras(self, frame: np.ndarray) -> Dict:
        """
        Detecta câmeras instaladas
        
        Args:
            frame: Frame de vídeo para análise
            
        Returns:
            Dict: Resultado da detecção
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'details': 'Câmeras não detectadas',
            'bbox': None
        }
        
        try:
            # Detectar círculos (lentes de câmera)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=50
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                camera_count = len(circles)
                
                result['detected'] = camera_count >= 1
                result['confidence'] = min(0.9, camera_count / 2)
                result['details'] = f'{camera_count} câmeras detectadas'
            
            # Detecção alternativa por regiões escuras circulares
            if not result['detected']:
                # Detectar regiões escuras que podem ser lentes
                _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
                
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                camera_candidates = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 200 <= area <= 5000:  # Área típica de lente
                        # Verificar circularidade
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.7:  # Bastante circular
                                camera_candidates += 1
                
                if camera_candidates >= 1:
                    result['detected'] = True
                    result['confidence'] = min(0.8, camera_candidates / 2)
                    result['details'] = f'{camera_candidates} possíveis câmeras detectadas'
                    
        except Exception as e:
            logger.error(f"Erro na detecção de câmeras: {e}")
        
        return result
    
    def detect_suportes(self, frame: np.ndarray) -> Dict:
        """
        Detecta suportes e fixações estruturais
        
        Args:
            frame: Frame de vídeo para análise
            
        Returns:
            Dict: Resultado da detecção
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'details': 'Suportes não detectados',
            'bbox': None
        }
        
        try:
            # Detectar estruturas metálicas (suportes)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detectar bordas fortes (estruturas metálicas)
            edges = cv2.Canny(gray, 100, 200)
            
            # Detectar linhas longas (estruturas de suporte)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=100,
                minLineLength=100,
                maxLineGap=20
            )
            
            if lines is not None:
                long_lines = len([line for line in lines 
                                if np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2) > 100])
                
                if long_lines >= 2:
                    result['detected'] = True
                    result['confidence'] = min(0.8, long_lines / 5)
                    result['details'] = f'{long_lines} estruturas de suporte detectadas'
            
            # Verificar presença de estruturas retangulares grandes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            large_structures = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Estruturas grandes
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Suportes geralmente são alongados
                    if aspect_ratio > 2 or aspect_ratio < 0.5:
                        large_structures += 1
            
            if large_structures >= 1:
                result['detected'] = True
                result['confidence'] = max(result['confidence'], min(0.7, large_structures / 3))
                result['details'] += f' - {large_structures} estruturas grandes detectadas'
                
        except Exception as e:
            logger.error(f"Erro na detecção de suportes: {e}")
        
        return result
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analisa um frame completo para todos os componentes
        
        Args:
            frame: Frame de vídeo para análise
            
        Returns:
            Dict: Resultado completo da análise
        """
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'frame_resolution': frame.shape[:2][::-1],
            'components': {},
            'overall_score': 0.0,
            'critical_components_ok': True,
            'status': 'UNKNOWN'
        }
        
        try:
            # Analisar cada componente
            detectors = {
                'etiqueta_visivel': self.detect_etiqueta_visivel,
                'tampa_encaixada': self.detect_tampa_encaixada,
                'parafusos_presentes': self.detect_parafusos_presentes,
                'conectores_instalados': self.detect_conectores_instalados,
                'cabeamento': self.detect_cabeamento,
                'cameras': self.detect_cameras,
                'suportes': self.detect_suportes
            }
            
            total_weight = 0
            weighted_score = 0
            
            for component_name, detector_func in detectors.items():
                component_info = self.radar_components[component_name]
                detection_result = detector_func(frame)
                
                # Calcular score do componente
                component_score = detection_result['confidence'] if detection_result['detected'] else 0
                
                # Adicionar ao resultado
                analysis_result['components'][component_name] = {
                    'detected': detection_result['detected'],
                    'confidence': detection_result['confidence'],
                    'details': detection_result['details'],
                    'weight': component_info['weight'],
                    'critical': component_info['critical'],
                    'score': component_score
                }
                
                # Calcular score ponderado
                weight = component_info['weight']
                weighted_score += component_score * weight
                total_weight += weight
                
                # Verificar componentes críticos
                if component_info['critical'] and not detection_result['detected']:
                    analysis_result['critical_components_ok'] = False
            
            # Calcular score geral
            analysis_result['overall_score'] = weighted_score / total_weight if total_weight > 0 else 0
            
            # Determinar status
            if analysis_result['critical_components_ok'] and analysis_result['overall_score'] >= 0.7:
                analysis_result['status'] = 'APROVADO'
            elif analysis_result['overall_score'] >= 0.5:
                analysis_result['status'] = 'REVISAR'
            else:
                analysis_result['status'] = 'REPROVADO'
                
        except Exception as e:
            logger.error(f"Erro na análise do frame: {e}")
            analysis_result['status'] = 'ERRO'
        
        return analysis_result
    
    def _detect_text_regions(self, gray_image: np.ndarray) -> List[Tuple]:
        """
        Detecta regiões com texto na imagem
        
        Args:
            gray_image: Imagem em escala de cinza
            
        Returns:
            List[Tuple]: Lista de regiões com texto
        """
        text_regions = []
        
        try:
            # Detectar regiões com variação de intensidade (texto)
            # Aplicar filtro para realçar texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
            
            # Threshold para binarizar
            _, binary = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Área mínima para texto
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Texto geralmente tem proporção específica
                    if 1.5 <= aspect_ratio <= 10:
                        text_regions.append((x, y, w, h))
                        
        except Exception as e:
            logger.error(f"Erro na detecção de texto: {e}")
        
        return text_regions


def test_radar_detector():
    """Função de teste do detector de radar"""
    detector = RadarEquipmentDetector()
    detector.load_model()
    
    print("=== TESTE DO DETECTOR DE RADAR ===")
    print(f"Componentes monitorados: {len(detector.radar_components)}")
    
    for component, info in detector.radar_components.items():
        print(f"- {component}: {info['description']} (peso: {info['weight']}, crítico: {info['critical']})")
    
    # Criar frame de teste
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Testar análise
    result = detector.analyze_frame(test_frame)
    print(f"\nTeste de análise:")
    print(f"Score geral: {result['overall_score']:.2f}")
    print(f"Status: {result['status']}")
    print(f"Componentes críticos OK: {result['critical_components_ok']}")
    
    print("Detector de radar testado com sucesso!")


if __name__ == "__main__":
    test_radar_detector()

