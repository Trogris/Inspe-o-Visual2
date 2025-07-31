#!/usr/bin/env python3
"""
Sistema de Armazenamento de Dados
Sistema de Verificação Visual Automatizada

Este módulo gerencia o armazenamento de dados de inspeção,
incluindo banco SQLite, organização de imagens e relatórios.
"""

import sqlite3
import os
import json
import shutil
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import cv2
import numpy as np
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataStorage:
    """
    Classe para gerenciar armazenamento de dados de inspeção
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Inicializa o sistema de armazenamento
        
        Args:
            data_dir: Diretório base para armazenamento
        """
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "inspection_database.db")
        self.images_dir = os.path.join(data_dir, "images")
        self.reports_dir = os.path.join(data_dir, "reports")
        
        # Criar diretórios se não existirem
        self._create_directories()
        
        # Inicializar banco de dados
        self._initialize_database()
        
        logger.info(f"Sistema de armazenamento inicializado em: {data_dir}")
    
    def _create_directories(self):
        """
        Cria estrutura de diretórios necessária
        """
        directories = [
            self.data_dir,
            self.images_dir,
            self.reports_dir,
            os.path.join(self.images_dir, "originals"),
            os.path.join(self.images_dir, "processed"),
            os.path.join(self.images_dir, "thumbnails")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Estrutura de diretórios criada")
    
    def _initialize_database(self):
        """
        Inicializa banco de dados SQLite com tabelas necessárias
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tabela principal de inspeções
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS inspections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        op_number TEXT,
                        operator_name TEXT,
                        equipment_code TEXT,
                        overall_status TEXT,
                        confidence_score REAL,
                        image_original_path TEXT,
                        image_processed_path TEXT,
                        notes TEXT,
                        created_date DATE DEFAULT (DATE('now'))
                    )
                ''')
                
                # Tabela de detecções visuais
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS visual_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        inspection_id INTEGER,
                        detection_type TEXT,
                        class_name TEXT,
                        confidence REAL,
                        bbox_x1 INTEGER,
                        bbox_y1 INTEGER,
                        bbox_x2 INTEGER,
                        bbox_y2 INTEGER,
                        status TEXT,
                        FOREIGN KEY (inspection_id) REFERENCES inspections (id)
                    )
                ''')
                
                # Tabela de dados OCR
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ocr_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        inspection_id INTEGER,
                        text_detected TEXT,
                        confidence REAL,
                        bbox_x INTEGER,
                        bbox_y INTEGER,
                        bbox_w INTEGER,
                        bbox_h INTEGER,
                        extracted_info TEXT,  -- JSON com informações estruturadas
                        FOREIGN KEY (inspection_id) REFERENCES inspections (id)
                    )
                ''')
                
                # Tabela de checklist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS checklist_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        inspection_id INTEGER,
                        item_name TEXT,
                        status TEXT,
                        confidence REAL,
                        details TEXT,  -- JSON com detalhes adicionais
                        FOREIGN KEY (inspection_id) REFERENCES inspections (id)
                    )
                ''')
                
                # Tabela de configurações
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_config (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Banco de dados inicializado com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
    
    def save_inspection(self, inspection_data: Dict) -> Optional[int]:
        """
        Salva dados completos de uma inspeção
        
        Args:
            inspection_data: Dados da inspeção
            
        Returns:
            int: ID da inspeção salva ou None se erro
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Extrair dados principais
                main_data = inspection_data.get('main', {})
                visual_data = inspection_data.get('visual_analysis', {})
                ocr_data = inspection_data.get('ocr_analysis', {})
                
                # Inserir inspeção principal
                cursor.execute('''
                    INSERT INTO inspections (
                        op_number, operator_name, equipment_code, overall_status,
                        confidence_score, image_original_path, image_processed_path, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    main_data.get('op_number'),
                    main_data.get('operator_name'),
                    main_data.get('equipment_code'),
                    visual_data.get('overall_status'),
                    visual_data.get('confidence_score', 0.0),
                    main_data.get('image_original_path'),
                    main_data.get('image_processed_path'),
                    main_data.get('notes', '')
                ))
                
                inspection_id = cursor.lastrowid
                
                # Salvar detecções visuais
                for detection in visual_data.get('detections', []):
                    bbox = detection.get('bbox', {})
                    cursor.execute('''
                        INSERT INTO visual_detections (
                            inspection_id, detection_type, class_name, confidence,
                            bbox_x1, bbox_y1, bbox_x2, bbox_y2, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        inspection_id, 'yolo', detection.get('class_name'),
                        detection.get('confidence'), bbox.get('x1'), bbox.get('y1'),
                        bbox.get('x2'), bbox.get('y2'), 'DETECTED'
                    ))
                
                # Salvar dados OCR
                for text_detection in ocr_data.get('text_detections', []):
                    bbox = text_detection.get('bbox', {})
                    cursor.execute('''
                        INSERT INTO ocr_data (
                            inspection_id, text_detected, confidence,
                            bbox_x, bbox_y, bbox_w, bbox_h, extracted_info
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        inspection_id, text_detection.get('text'),
                        text_detection.get('confidence'),
                        bbox.get('x1'), bbox.get('y1'),
                        bbox.get('x2', 0) - bbox.get('x1', 0),
                        bbox.get('y2', 0) - bbox.get('y1', 0),
                        json.dumps(ocr_data.get('equipment_info', {}))
                    ))
                
                # Salvar checklist
                for item_name, item_data in visual_data.get('checklist', {}).items():
                    cursor.execute('''
                        INSERT INTO checklist_items (
                            inspection_id, item_name, status, confidence, details
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (
                        inspection_id, item_name, item_data.get('status'),
                        item_data.get('confidence', 0.0),
                        json.dumps(item_data.get('details', {}))
                    ))
                
                conn.commit()
                logger.info(f"Inspeção salva com ID: {inspection_id}")
                return inspection_id
                
        except Exception as e:
            logger.error(f"Erro ao salvar inspeção: {e}")
            return None
    
    def save_image_with_metadata(self, image: np.ndarray, metadata: Dict, 
                                image_type: str = "original") -> Optional[str]:
        """
        Salva imagem com metadados organizados por data
        
        Args:
            image: Imagem a ser salva
            metadata: Metadados da imagem
            image_type: Tipo da imagem ('original', 'processed', 'thumbnail')
            
        Returns:
            str: Caminho da imagem salva ou None se erro
        """
        try:
            # Criar estrutura de diretórios por data
            today = date.today()
            date_dir = os.path.join(self.images_dir, image_type, today.strftime("%Y-%m-%d"))
            os.makedirs(date_dir, exist_ok=True)
            
            # Gerar nome do arquivo
            timestamp = datetime.now().strftime("%H%M%S")
            op_number = metadata.get('op_number', 'unknown')
            filename = f"{op_number}_{timestamp}.jpg"
            filepath = os.path.join(date_dir, filename)
            
            # Salvar imagem
            cv2.imwrite(filepath, image)
            
            # Salvar metadados em arquivo JSON
            metadata_file = filepath.replace('.jpg', '_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Imagem salva: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Erro ao salvar imagem: {e}")
            return None
    
    def create_thumbnail(self, image_path: str, max_size: Tuple[int, int] = (300, 300)) -> Optional[str]:
        """
        Cria thumbnail de uma imagem
        
        Args:
            image_path: Caminho da imagem original
            max_size: Tamanho máximo do thumbnail
            
        Returns:
            str: Caminho do thumbnail ou None se erro
        """
        try:
            # Carregar imagem
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Calcular novo tamanho mantendo proporção
            h, w = image.shape[:2]
            max_w, max_h = max_size
            
            if w > h:
                new_w = max_w
                new_h = int(h * (max_w / w))
            else:
                new_h = max_h
                new_w = int(w * (max_h / h))
            
            # Redimensionar
            thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Determinar caminho do thumbnail
            base_name = os.path.basename(image_path)
            date_str = datetime.now().strftime("%Y-%m-%d")
            thumb_dir = os.path.join(self.images_dir, "thumbnails", date_str)
            os.makedirs(thumb_dir, exist_ok=True)
            
            thumb_path = os.path.join(thumb_dir, f"thumb_{base_name}")
            
            # Salvar thumbnail
            cv2.imwrite(thumb_path, thumbnail)
            
            return thumb_path
            
        except Exception as e:
            logger.error(f"Erro ao criar thumbnail: {e}")
            return None
    
    def get_inspections_by_date(self, target_date: date = None) -> List[Dict]:
        """
        Obtém inspeções por data
        
        Args:
            target_date: Data alvo (hoje se None)
            
        Returns:
            List[Dict]: Lista de inspeções
        """
        if target_date is None:
            target_date = date.today()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Para acessar por nome da coluna
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM inspections 
                    WHERE created_date = ?
                    ORDER BY timestamp DESC
                ''', (target_date,))
                
                inspections = []
                for row in cursor.fetchall():
                    inspection = dict(row)
                    
                    # Buscar checklist items
                    cursor.execute('''
                        SELECT * FROM checklist_items 
                        WHERE inspection_id = ?
                    ''', (inspection['id'],))
                    
                    checklist = {}
                    for item_row in cursor.fetchall():
                        item_dict = dict(item_row)
                        checklist[item_dict['item_name']] = {
                            'status': item_dict['status'],
                            'confidence': item_dict['confidence'],
                            'details': json.loads(item_dict['details'] or '{}')
                        }
                    
                    inspection['checklist'] = checklist
                    inspections.append(inspection)
                
                return inspections
                
        except Exception as e:
            logger.error(f"Erro ao buscar inspeções: {e}")
            return []
    
    def get_inspection_by_id(self, inspection_id: int) -> Optional[Dict]:
        """
        Obtém inspeção completa por ID
        
        Args:
            inspection_id: ID da inspeção
            
        Returns:
            Dict: Dados completos da inspeção ou None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Buscar inspeção principal
                cursor.execute('SELECT * FROM inspections WHERE id = ?', (inspection_id,))
                inspection_row = cursor.fetchone()
                
                if not inspection_row:
                    return None
                
                inspection = dict(inspection_row)
                
                # Buscar detecções visuais
                cursor.execute('''
                    SELECT * FROM visual_detections WHERE inspection_id = ?
                ''', (inspection_id,))
                inspection['visual_detections'] = [dict(row) for row in cursor.fetchall()]
                
                # Buscar dados OCR
                cursor.execute('''
                    SELECT * FROM ocr_data WHERE inspection_id = ?
                ''', (inspection_id,))
                ocr_rows = cursor.fetchall()
                inspection['ocr_data'] = []
                for row in ocr_rows:
                    ocr_dict = dict(row)
                    ocr_dict['extracted_info'] = json.loads(ocr_dict['extracted_info'] or '{}')
                    inspection['ocr_data'].append(ocr_dict)
                
                # Buscar checklist
                cursor.execute('''
                    SELECT * FROM checklist_items WHERE inspection_id = ?
                ''', (inspection_id,))
                checklist = {}
                for row in cursor.fetchall():
                    item_dict = dict(row)
                    checklist[item_dict['item_name']] = {
                        'status': item_dict['status'],
                        'confidence': item_dict['confidence'],
                        'details': json.loads(item_dict['details'] or '{}')
                    }
                inspection['checklist'] = checklist
                
                return inspection
                
        except Exception as e:
            logger.error(f"Erro ao buscar inspeção {inspection_id}: {e}")
            return None
    
    def generate_daily_report(self, target_date: date = None) -> Optional[str]:
        """
        Gera relatório diário em CSV
        
        Args:
            target_date: Data do relatório (hoje se None)
            
        Returns:
            str: Caminho do arquivo de relatório ou None
        """
        if target_date is None:
            target_date = date.today()
        
        try:
            inspections = self.get_inspections_by_date(target_date)
            
            if not inspections:
                logger.info(f"Nenhuma inspeção encontrada para {target_date}")
                return None
            
            # Preparar dados para DataFrame
            report_data = []
            for inspection in inspections:
                # Contar itens do checklist
                checklist = inspection.get('checklist', {})
                total_items = len(checklist)
                approved_items = sum(1 for item in checklist.values() if item['status'] == 'OK')
                
                row = {
                    'ID': inspection['id'],
                    'Timestamp': inspection['timestamp'],
                    'OP': inspection['op_number'],
                    'Operador': inspection['operator_name'],
                    'Código Equipamento': inspection['equipment_code'],
                    'Status Geral': inspection['overall_status'],
                    'Confiança': f"{inspection['confidence_score']:.2f}",
                    'Itens Checklist': f"{approved_items}/{total_items}",
                    'Observações': inspection['notes']
                }
                
                # Adicionar status individual dos itens
                for item_name, item_data in checklist.items():
                    row[f"Status_{item_name}"] = item_data['status']
                
                report_data.append(row)
            
            # Criar DataFrame
            df = pd.DataFrame(report_data)
            
            # Salvar relatório
            report_filename = f"relatorio_diario_{target_date.strftime('%Y%m%d')}.csv"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            df.to_csv(report_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"Relatório diário gerado: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório diário: {e}")
            return None
    
    def get_statistics(self, start_date: date = None, end_date: date = None) -> Dict:
        """
        Obtém estatísticas das inspeções
        
        Args:
            start_date: Data inicial (última semana se None)
            end_date: Data final (hoje se None)
            
        Returns:
            Dict: Estatísticas
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = date.fromordinal(end_date.toordinal() - 7)  # Última semana
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total de inspeções
                cursor.execute('''
                    SELECT COUNT(*) FROM inspections 
                    WHERE created_date BETWEEN ? AND ?
                ''', (start_date, end_date))
                total_inspections = cursor.fetchone()[0]
                
                # Inspeções aprovadas
                cursor.execute('''
                    SELECT COUNT(*) FROM inspections 
                    WHERE created_date BETWEEN ? AND ? AND overall_status = 'APROVADO'
                ''', (start_date, end_date))
                approved_inspections = cursor.fetchone()[0]
                
                # Taxa de aprovação
                approval_rate = (approved_inspections / total_inspections * 100) if total_inspections > 0 else 0
                
                # Estatísticas por operador
                cursor.execute('''
                    SELECT operator_name, COUNT(*) as total,
                           SUM(CASE WHEN overall_status = 'APROVADO' THEN 1 ELSE 0 END) as approved
                    FROM inspections 
                    WHERE created_date BETWEEN ? AND ?
                    GROUP BY operator_name
                ''', (start_date, end_date))
                
                operator_stats = []
                for row in cursor.fetchall():
                    operator_stats.append({
                        'operator': row[0],
                        'total': row[1],
                        'approved': row[2],
                        'approval_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0
                    })
                
                # Problemas mais comuns
                cursor.execute('''
                    SELECT item_name, COUNT(*) as failures
                    FROM checklist_items 
                    WHERE status = 'MISSING'
                    GROUP BY item_name
                    ORDER BY failures DESC
                    LIMIT 5
                ''')
                
                common_issues = [{'item': row[0], 'count': row[1]} for row in cursor.fetchall()]
                
                statistics = {
                    'period': {
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    },
                    'totals': {
                        'total_inspections': total_inspections,
                        'approved_inspections': approved_inspections,
                        'rejected_inspections': total_inspections - approved_inspections,
                        'approval_rate': approval_rate
                    },
                    'operator_stats': operator_stats,
                    'common_issues': common_issues
                }
                
                return statistics
                
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Remove dados antigos para economizar espaço
        
        Args:
            days_to_keep: Dias de dados para manter
        """
        try:
            cutoff_date = date.fromordinal(date.today().toordinal() - days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Buscar inspeções antigas
                cursor.execute('''
                    SELECT id, image_original_path, image_processed_path 
                    FROM inspections WHERE created_date < ?
                ''', (cutoff_date,))
                
                old_inspections = cursor.fetchall()
                
                # Remover arquivos de imagem
                for inspection in old_inspections:
                    for image_path in [inspection[1], inspection[2]]:
                        if image_path and os.path.exists(image_path):
                            os.remove(image_path)
                
                # Remover dados do banco
                cursor.execute('DELETE FROM checklist_items WHERE inspection_id IN (SELECT id FROM inspections WHERE created_date < ?)', (cutoff_date,))
                cursor.execute('DELETE FROM ocr_data WHERE inspection_id IN (SELECT id FROM inspections WHERE created_date < ?)', (cutoff_date,))
                cursor.execute('DELETE FROM visual_detections WHERE inspection_id IN (SELECT id FROM inspections WHERE created_date < ?)', (cutoff_date,))
                cursor.execute('DELETE FROM inspections WHERE created_date < ?', (cutoff_date,))
                
                conn.commit()
                
                logger.info(f"Limpeza concluída: {len(old_inspections)} inspeções antigas removidas")
                
        except Exception as e:
            logger.error(f"Erro na limpeza de dados: {e}")


def test_data_storage():
    """
    Função de teste para o sistema de armazenamento
    """
    print("=== Teste do Sistema de Armazenamento ===")
    
    # Criar instância
    storage = DataStorage()
    
    # Dados de teste
    test_inspection = {
        'main': {
            'op_number': 'OP-2025-001',
            'operator_name': 'João Silva',
            'equipment_code': 'DNITMS-FSCII-7480',
            'notes': 'Teste do sistema'
        },
        'visual_analysis': {
            'overall_status': 'APROVADO',
            'confidence_score': 0.85,
            'detections': [
                {
                    'class_name': 'etiqueta_principal',
                    'confidence': 0.9,
                    'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 150}
                }
            ],
            'checklist': {
                'estrutura_metalica': {'status': 'OK', 'confidence': 0.8},
                'led_status': {'status': 'OK', 'confidence': 0.9},
                'etiquetas': {'status': 'OK', 'confidence': 0.85}
            }
        },
        'ocr_analysis': {
            'text_detections': [
                {
                    'text': 'DNITMS-FSCII-7480',
                    'confidence': 0.95,
                    'bbox': {'x1': 110, 'y1': 110, 'x2': 190, 'y2': 140}
                }
            ],
            'equipment_info': {
                'equipment_code': 'DNITMS-FSCII-7480',
                'road_info': 'BR-262 km 590+630'
            }
        }
    }
    
    # Salvar inspeção de teste
    inspection_id = storage.save_inspection(test_inspection)
    if inspection_id:
        print(f"Inspeção de teste salva com ID: {inspection_id}")
        
        # Buscar inspeção salva
        saved_inspection = storage.get_inspection_by_id(inspection_id)
        if saved_inspection:
            print(f"Inspeção recuperada: {saved_inspection['equipment_code']}")
        
        # Gerar relatório diário
        report_path = storage.generate_daily_report()
        if report_path:
            print(f"Relatório diário gerado: {report_path}")
        
        # Obter estatísticas
        stats = storage.get_statistics()
        print(f"Estatísticas: {stats['totals']['total_inspections']} inspeções")
        
    else:
        print("Falha ao salvar inspeção de teste")


if __name__ == "__main__":
    test_data_storage()

