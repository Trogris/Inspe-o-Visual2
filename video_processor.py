#!/usr/bin/env python3
"""
Processador de Vídeo - Versão Corrigida
Extração robusta de frames com tratamento de erros
"""

import cv2
import numpy as np
import tempfile
import os
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Processador de vídeo para extração de frames
    """
    
    def __init__(self):
        """Inicializa o processador de vídeo"""
        self.supported_formats = ['mp4', 'mov', 'avi', 'mkv', 'wmv', 'mpeg4']
        self.max_size_mb = 200  # Aumentado para 200MB
        logger.info("Processador de vídeo inicializado")
    
    def validate_video_file(self, uploaded_file) -> bool:
        """
        Valida arquivo de vídeo
        
        Args:
            uploaded_file: Arquivo enviado pelo Streamlit
            
        Returns:
            bool: True se válido, False caso contrário
        """
        try:
            if not uploaded_file:
                return False
            
            # Verificar tamanho
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > self.max_size_mb:
                logger.error(f"Arquivo muito grande: {file_size_mb:.1f}MB (máximo: {self.max_size_mb}MB)")
                return False
            
            # Verificar extensão
            file_extension = uploaded_file.name.lower().split('.')[-1]
            if file_extension not in self.supported_formats:
                logger.error(f"Formato não suportado: {file_extension}")
                return False
            
            logger.info(f"Arquivo válido: {uploaded_file.name} ({uploaded_file.size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação: {e}")
            return False
    
    def get_video_info(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """
        Obtém informações básicas do vídeo
        
        Args:
            uploaded_file: Arquivo de vídeo
            
        Returns:
            Dict com informações ou None se erro
        """
        temp_path = None
        try:
            # Salvar arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_path = temp_file.name
                temp_file.write(uploaded_file.read())
            
            # Abrir vídeo
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                logger.error("Não foi possível abrir o vídeo")
                return None
            
            # Obter informações básicas
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            size_mb = uploaded_file.size / (1024 * 1024)
            
            cap.release()
            
            return {
                'filename': uploaded_file.name,
                'size_mb': size_mb,
                'duration': duration,
                'fps': fps,
                'total_frames': frame_count,
                'resolution': (width, height),
                'format': os.path.splitext(uploaded_file.name)[1].lower()
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter informações do vídeo: {e}")
            return None
            
        finally:
            # Limpar arquivo temporário
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def extract_frames_from_video(self, uploaded_file, num_frames: int = 10) -> List[np.ndarray]:
        """
        Extrai frames distribuídos uniformemente do vídeo
        
        Args:
            uploaded_file: Arquivo de vídeo
            num_frames: Número de frames a extrair
            
        Returns:
            Lista de frames como arrays numpy
        """
        temp_path = None
        frames = []
        
        try:
            # Salvar arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_path = temp_file.name
                # Reset file pointer
                uploaded_file.seek(0)
                temp_file.write(uploaded_file.read())
            
            # Abrir vídeo
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                logger.error("Não foi possível abrir o vídeo para extração")
                return []
            
            # Obter informações do vídeo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0:
                logger.error("Vídeo não contém frames válidos")
                cap.release()
                return []
            
            logger.info(f"Vídeo: {total_frames} frames, {fps:.2f} FPS")
            
            # Calcular posições dos frames
            if total_frames < num_frames:
                # Se o vídeo tem menos frames que o solicitado, usar todos
                frame_positions = list(range(total_frames))
            else:
                # Distribuir uniformemente
                step = max(1, total_frames // num_frames)
                frame_positions = [i * step for i in range(num_frames)]
                # Garantir que não exceda o total
                frame_positions = [min(pos, total_frames - 1) for pos in frame_positions]
            
            # Extrair frames
            for i, frame_pos in enumerate(frame_positions):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Redimensionar se muito grande (otimização)
                    height, width = frame.shape[:2]
                    if width > 1920 or height > 1080:
                        # Redimensionar mantendo proporção
                        scale = min(1920/width, 1080/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    frames.append(frame)
                    time_seconds = frame_pos / fps if fps > 0 else 0
                    logger.info(f"Frame {i+1}/{len(frame_positions)} extraído (posição: {frame_pos}, tempo: {time_seconds:.2f}s)")
                else:
                    logger.warning(f"Falha ao extrair frame na posição {frame_pos}")
            
            cap.release()
            
            if frames:
                logger.info(f"Extração concluída: {len(frames)} frames extraídos")
            else:
                logger.error("Nenhum frame foi extraído com sucesso")
            
            return frames
            
        except Exception as e:
            logger.error(f"Erro na extração de frames: {e}")
            return []
            
        finally:
            # Limpar arquivo temporário
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """
        Valida se um frame é válido
        
        Args:
            frame: Frame como array numpy
            
        Returns:
            bool: True se válido
        """
        try:
            if frame is None:
                return False
            
            if not isinstance(frame, np.ndarray):
                return False
            
            if len(frame.shape) != 3:
                return False
            
            height, width, channels = frame.shape
            if height < 100 or width < 100 or channels != 3:
                return False
            
            return True
            
        except:
            return False

