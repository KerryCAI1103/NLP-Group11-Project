#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感氛围电影推荐系统
整合语义搜索与情感分析，实现基于心情的电影推荐

功能特点：
1. 使用Qwen3-Embedding进行语义相似度计算
2. 基于情感向量进行个性化推荐
3. 支持混合推荐（语义+情感）
4. 可视化情感分析结果
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
import config  # 导入配置文件
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class EmotionMovieRecommender:
    """情感氛围电影推荐系统"""
    
    def __init__(self, model_name: str = None):
        """
        初始化推荐系统
        
        参数:
            model_name: 使用的嵌入模型名称，默认为None则使用config配置
        """
        print("=" * 80)
        print("🎭 情感氛围电影推荐系统 - 初始化中...")
        print("=" * 80)
        
        # 定义情感标签映射（中文->英文）
        self.emotion_mapping = {
            'joy': 'joy',
            'sadness': 'sadness', 
            'anger': 'anger',
            'fear': 'fear',
            'love': 'love',
            'hope': 'hope',
            'loneliness': 'loneliness',
            'inspiration': 'inspiration',
            'tension': 'tension',
            'peace': 'peace',
            # 中文映射
            '快乐': 'joy', '开心': 'joy', '高兴': 'joy', '愉快': 'joy', '欢乐': 'joy',
            '悲伤': 'sadness', '难过': 'sadness', '伤心': 'sadness', '忧郁': 'sadness',
            '愤怒': 'anger', '生气': 'anger', '怒火': 'anger',
            '恐惧': 'fear', '害怕': 'fear', '恐怖': 'fear', '惊吓': 'fear',
            '爱': 'love', '爱情': 'love', '浪漫': 'love', '甜蜜': 'love',
            '希望': 'hope', '期望': 'hope', '期待': 'hope', '梦想': 'hope',
            '孤独': 'loneliness', '孤单': 'loneliness', '寂寞': 'loneliness',
            '励志': 'inspiration', '鼓舞': 'inspiration', '激励': 'inspiration',
            '紧张': 'tension', '刺激': 'tension', '悬疑': 'tension', '惊悚': 'tension',
            '平静': 'peace', '安宁': 'peace', '宁静': 'peace', '祥和': 'peace'
        }
        
        # 固定的10种情感维度（与爬虫中定义的情感词典一致）
        self.fixed_emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'love', 
                                   'hope', 'loneliness', 'inspiration', 'tension', 'peace']
        
        # 加载嵌入模型
        if model_name is None:
            # 使用config文件的load_model函数（自动使用ModelScope镜像）
            self.model = config.load_model(device='cpu')
        else:
            print(f"加载指定模型: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
        print("✓ 嵌入模型加载成功!")
        
        # 初始化数据结构
        self.movies = []  # 原始电影数据
        self.movie_texts = []  # 用于嵌入的文本
        self.semantic_embeddings = None  # 语义嵌入向量
        self.emotion_vectors = None  # 情感向量矩阵
        self.emotion_labels = []  # 情感标签列表
        self.emotion_profiles = {}  # 每部电影的情感分布
        
    def load_movies_from_json(self, json_path: str) -> List[Dict]:
        """
        从JSON文件加载电影情感数据
        
        参数:
            json_path: JSON文件路径
            
        返回:
            List[Dict]: 电影数据列表
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✓ 成功从 {json_path} 加载 {len(data)} 部电影的情感数据")
            
            # 转换为标准格式
            formatted_movies = []
            for movie in data:
                # 检查情感分布数据
                emotion_profile = movie.get('emotion_profile', {})
                
                # 确保情感向量包含所有10个维度，缺失的补0
                fixed_emotion_profile = {}
                for emotion in self.fixed_emotion_labels:
                    fixed_emotion_profile[emotion] = emotion_profile.get(emotion, 0.0)
                
                # 归一化情感向量
                total = sum(fixed_emotion_profile.values())
                if total > 0:
                    for emotion in fixed_emotion_profile:
                        fixed_emotion_profile[emotion] = round(fixed_emotion_profile[emotion] / total, 3)
                
                # 构建标准电影记录
                movie_record = {
                    'id': movie.get('id', ''),
                    'title': movie.get('title', ''),
                    'original_title': movie.get('original_title', ''),
                    'year': movie.get('year', ''),
                    'plot': movie.get('plot', ''),
                    'tagline': movie.get('tagline', ''),
                    'genres': movie.get('genres', []),
                    'rating': movie.get('rating', 0),
                    'runtime': movie.get('runtime', 0),
                    
                    # 使用修正后的情感数据
                    'emotion_profile': fixed_emotion_profile,
                    'mood_tags': movie.get('mood_tags', []),
                    'dominant_emotions': movie.get('dominant_emotions', []),
                    'emotional_complexity': movie.get('emotional_complexity', 0),
                    
                    # 源数据
                    'source': movie.get('source', 'unknown')
                }
                formatted_movies.append(movie_record)
            
            # 显示情感向量统计
            emotion_stats = {}
            for movie in formatted_movies[:5]:  # 只检查前5部
                profile = movie.get('emotion_profile', {})
                print(f"  《{movie['title']}》情感向量维度: {len(profile)}, 样本值: {list(profile.items())[:3]}...")
                
                for emotion, value in profile.items():
                    if value > 0:
                        emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
            
            print(f"情感分布统计: {len(emotion_stats)}种情感被使用")
            
            return formatted_movies
            
        except Exception as e:
            print(f"✗ 加载JSON文件失败: {e}")
            print("将使用示例电影数据...")
            return self.load_sample_movies()
    
    def load_movies_from_csv(self, csv_path: str) -> List[Dict]:
        """
        从CSV文件加载电影数据（兼容原有格式）
        
        参数:
            csv_path: CSV文件路径
            
        返回:
            List[Dict]: 电影数据列表
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ 成功从 {csv_path} 加载 {len(df)} 部电影")
            
            # 转换为标准格式
            movies = []
            for _, row in df.iterrows():
                # 解析情感向量（如果存在）
                emotion_profile = {}
                emotion_vector_str = str(row.get('emotion_vector', ''))
                if emotion_vector_str and ':' in emotion_vector_str:
                    for pair in emotion_vector_str.split('|'):
                        if ':' in pair:
                            emotion, value = pair.split(':', 1)
                            try:
                                emotion_profile[emotion.strip()] = float(value.strip())
                            except:
                                pass
                
                # 如果情感向量为空，创建一个默认向量
                if not emotion_profile:
                    for emotion in self.fixed_emotion_labels:
                        emotion_profile[emotion] = 0.0
                
                movie = {
                    'id': str(row.get('movie_id', '')),
                    'title': str(row.get('title', '')),
                    'plot': str(row.get('plot', '')),
                    'genres': str(row.get('genres', '')).split('|') if '|' in str(row.get('genres', '')) else [],
                    'year': str(row.get('year', '')),
                    'rating': float(row.get('rating', 0)),
                    
                    # 情感数据
                    'emotion_profile': emotion_profile,
                    'mood_tags': str(row.get('mood_tags', '')).split('|') if '|' in str(row.get('mood_tags', '')) else [],
                    'dominant_emotions': str(row.get('dominant_emotions', '')).split('|') if '|' in str(row.get('dominant_emotions', '')) else [],
                    'source': 'csv'
                }
                movies.append(movie)
            
            return movies
            
        except Exception as e:
            print(f"✗ 加载CSV文件失败: {e}")
            return []
    
    def load_sample_movies(self) -> List[Dict]:
        """
        加载示例电影数据（包含情感信息）
        
        返回:
            List[Dict]: 示例电影数据
        """
        print("使用示例电影数据（包含情感分析）...")
        
        # 使用固定情感标签创建示例数据
        sample_movies = []
        
        movie_templates = [
            {
                "id": "1",
                "title": "肖申克的救赎",
                "plot": "银行家安迪被冤枉杀害妻子及其情人，被判无期徒刑。在肖申克监狱中，他凭借自己的知识和智慧，不仅改善了狱友的生活，还策划了一场惊人的越狱。",
                "genres": ["剧情", "犯罪"],
                "year": "1994",
                "rating": 9.3,
                "emotion_profile": {"hope": 0.35, "inspiration": 0.25, "sadness": 0.15, "anger": 0.10, "loneliness": 0.10, "love": 0.05, "joy": 0.0, "fear": 0.0, "tension": 0.0, "peace": 0.0},
                "mood_tags": ["充满希望", "励志", "感人", "救赎"],
                "dominant_emotions": ["hope", "inspiration"]
            },
            {
                "id": "2",
                "title": "星际穿越",
                "plot": "地球环境恶化，一组宇航员穿越虫洞为人类寻找新的家园，探索时间与爱的真谛。",
                "genres": ["科幻", "冒险", "剧情"],
                "year": "2014",
                "rating": 9.2,
                "emotion_profile": {"hope": 0.30, "love": 0.25, "wonder": 0.20, "loneliness": 0.15, "fear": 0.10, "sadness": 0.0, "anger": 0.0, "joy": 0.0, "inspiration": 0.0, "tension": 0.0, "peace": 0.0},
                "mood_tags": ["宏大叙事", "感人至深", "科幻史诗", "父女情深"],
                "dominant_emotions": ["hope", "love"]
            },
            {
                "id": "3",
                "title": "美丽人生",
                "plot": "犹太青年圭多与儿子被关进集中营，他用游戏的方式保护儿子的童心，展现了父爱的伟大。",
                "genres": ["剧情", "喜剧", "爱情"],
                "year": "1997",
                "rating": 9.5,
                "emotion_profile": {"love": 0.35, "hope": 0.25, "joy": 0.20, "sadness": 0.20, "fear": 0.0, "anger": 0.0, "loneliness": 0.0, "inspiration": 0.0, "tension": 0.0, "peace": 0.0},
                "mood_tags": ["感人至深", "父爱如山", "笑中带泪", "希望之光"],
                "dominant_emotions": ["love", "hope"]
            }
        ]
        
        for template in movie_templates:
            # 确保情感向量包含所有10个维度
            full_emotion_profile = {}
            for emotion in self.fixed_emotion_labels:
                full_emotion_profile[emotion] = template.get('emotion_profile', {}).get(emotion, 0.0)
            
            # 归一化
            total = sum(full_emotion_profile.values())
            if total > 0:
                for emotion in full_emotion_profile:
                    full_emotion_profile[emotion] = round(full_emotion_profile[emotion] / total, 3)
            
            movie = {
                'id': template['id'],
                'title': template['title'],
                'original_title': template.get('original_title', template['title']),
                'year': template['year'],
                'plot': template['plot'],
                'tagline': template.get('tagline', ''),
                'genres': template['genres'],
                'rating': template['rating'],
                'runtime': template.get('runtime', 120),
                'emotion_profile': full_emotion_profile,
                'mood_tags': template['mood_tags'],
                'dominant_emotions': template['dominant_emotions'],
                'emotional_complexity': len([v for v in full_emotion_profile.values() if v > 0.05]),
                'source': 'example'
            }
            sample_movies.append(movie)
        
        print(f"✓ 加载了 {len(sample_movies)} 部示例电影")
        return sample_movies
    
    def prepare_movie_texts(self, movies: List[Dict]) -> List[str]:
        """
        准备电影文本用于嵌入
        
        参数:
            movies: 电影数据列表
            
        返回:
            List[str]: 处理后的电影文本列表
        """
        texts = []
        for movie in movies:
            # 构建综合文本描述，包含语义信息和情感信息
            title = movie.get('title', '')
            plot = movie.get('plot', '')
            tagline = movie.get('tagline', '')
            genres = movie.get('genres', [])
            year = movie.get('year', '')
            mood_tags = movie.get('mood_tags', [])
            dominant_emotions = movie.get('dominant_emotions', [])
            
            # 组合所有信息
            text_parts = []
            
            # 1. 电影基本信息
            text_parts.append(f"电影《{title}》。")
            
            if tagline:
                text_parts.append(f"宣传语：{tagline}。")
            
            if plot:
                text_parts.append(f"剧情：{plot}")
            
            if genres:
                genres_text = "，".join(genres)
                text_parts.append(f"类型：{genres_text}。")
            
            if year:
                text_parts.append(f"年份：{year}。")
            
            # 2. 情感信息
            if mood_tags:
                mood_text = "，".join(mood_tags[:5])  # 最多5个情绪标签
                text_parts.append(f"情感氛围：{mood_text}。")
            
            if dominant_emotions:
                emotion_text = "，".join(dominant_emotions)
                text_parts.append(f"主导情感：{emotion_text}。")
            
            # 3. 情感描述（如果有详细情感分布）
            emotion_profile = movie.get('emotion_profile', {})
            if emotion_profile:
                # 只取前3个最强烈的情感
                top_emotions = sorted(emotion_profile.items(), key=lambda x: x[1], reverse=True)[:3]
                if top_emotions and any(score > 0 for _, score in top_emotions):
                    emotion_desc = "，".join([f"{emotion}({score:.2f})" for emotion, score in top_emotions])
                    text_parts.append(f"情感强度：{emotion_desc}。")
            
            # 合并所有部分
            full_text = " ".join(text_parts)
            texts.append(full_text)
        
        return texts
    
    def extract_emotion_vectors(self, movies: List[Dict]):
        """
        从电影数据中提取情感向量（修正版）
        
        参数:
            movies: 电影数据列表
        """
        print("提取电影情感向量...")
        
        # 使用固定的10种情感维度
        self.emotion_labels = self.fixed_emotion_labels.copy()
        print(f"使用固定的情感维度: {len(self.emotion_labels)} 种")
        
        # 构建情感向量矩阵
        num_movies = len(movies)
        num_emotions = len(self.emotion_labels)
        
        self.emotion_vectors = np.zeros((num_movies, num_emotions))
        self.emotion_profiles = {}
        
        for i, movie in enumerate(movies):
            movie_id = movie.get('id', i)
            emotion_profile = movie.get('emotion_profile', {})
            
            # 确保情感向量包含所有维度
            full_profile = {}
            for emotion in self.emotion_labels:
                # 从电影数据中获取情感值，如果没有则为0
                value = emotion_profile.get(emotion, 0.0)
                # 确保值在0-1之间
                full_profile[emotion] = max(0.0, min(1.0, float(value)))
            
            # 存储完整情感分布
            self.emotion_profiles[movie_id] = full_profile
            
            # 构建情感向量
            for j, emotion in enumerate(self.emotion_labels):
                self.emotion_vectors[i, j] = full_profile.get(emotion, 0.0)
        
        print(f"✓ 情感向量矩阵构建完成: {self.emotion_vectors.shape}")
        print(f"  情感向量样本（前3部电影）:")
        for i in range(min(3, len(movies))):
            non_zero = np.count_nonzero(self.emotion_vectors[i])
            print(f"    电影{i+1}: {non_zero}个非零值, 最大值: {self.emotion_vectors[i].max():.3f}")
    
    def index_movies(self, movies: List[Dict]):
        """
        索引电影数据，构建语义和情感向量
        
        参数:
            movies: 电影数据列表
        """
        print(f"\n开始索引 {len(movies)} 部电影...")
        
        # 存储原始电影数据
        self.movies = movies
        
        # 准备文本并构建语义嵌入
        print("1. 构建语义嵌入...")
        self.movie_texts = self.prepare_movie_texts(movies)
        
        self.semantic_embeddings = self.model.encode(
            self.movie_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        print(f"✓ 语义嵌入完成: {self.semantic_embeddings.shape}")
        
        # 提取情感向量
        print("2. 构建情感向量...")
        self.extract_emotion_vectors(movies)
        
        print("✓ 电影索引完成!")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        基于语义相似度搜索电影
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            
        返回:
            List[Tuple[Dict, float]]: 电影和相似度分数列表
        """
        if self.semantic_embeddings is None:
            raise ValueError("请先使用 index_movies() 方法索引电影")
        
        # 生成查询嵌入
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # 计算语义相似度
        semantic_similarities = util.cos_sim(query_embedding, self.semantic_embeddings)[0]
        
        # 获取top_k结果
        top_k = min(top_k, len(self.movies))
        top_results = semantic_similarities.topk(k=top_k)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            movie_data = self.movies[idx].copy()
            movie_data['semantic_score'] = score.item()
            movie_data['emotion_score'] = 0.0  # 语义搜索不考虑情感
            results.append((movie_data, score.item()))
        
        return results
    
    def emotion_search(self, target_emotions: Dict[str, float], top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        基于情感向量搜索电影（修正版）
        
        参数:
            target_emotions: 目标情感向量，格式为{情感: 强度}
            top_k: 返回结果数量
            
        返回:
            List[Tuple[Dict, float]]: 电影和相似度分数列表
        """
        if self.emotion_vectors is None:
            raise ValueError("请先使用 index_movies() 方法索引电影")
        
        # 转换情感标签为英文
        converted_emotions = {}
        for emotion, intensity in target_emotions.items():
            # 将中文情感标签映射为英文
            mapped_emotion = self.emotion_mapping.get(emotion, emotion)
            if mapped_emotion in self.emotion_labels:
                # 确保强度在合理范围内
                intensity_val = max(0.0, min(1.0, float(intensity)))
                converted_emotions[mapped_emotion] = intensity_val
            else:
                print(f"警告: 情感标签 '{emotion}' 不在情感词典中")
        
        # 构建目标情感向量
        target_vector = np.zeros(len(self.emotion_labels))
        for emotion, intensity in converted_emotions.items():
            if emotion in self.emotion_labels:
                idx = self.emotion_labels.index(emotion)
                target_vector[idx] = intensity
        
        # 归一化目标向量
        target_norm = np.linalg.norm(target_vector)
        if target_norm > 0:
            target_vector = target_vector / target_norm
        
        print(f"目标情感向量: {dict(zip(self.emotion_labels, target_vector.round(3)))}")
        
        # 计算情感相似度（余弦相似度）
        if np.all(target_vector == 0):
            print("警告: 目标情感向量全为零，无法计算相似度")
            return []
        
        # 计算情感相似度
        emotion_similarities = cosine_similarity([target_vector], self.emotion_vectors)[0]
        
        # 获取top_k结果
        top_k = min(top_k, len(self.movies))
        top_indices = emotion_similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            similarity = emotion_similarities[idx]
            movie_data = self.movies[idx].copy()
            movie_data['semantic_score'] = 0.0  # 情感搜索不考虑语义
            movie_data['emotion_score'] = similarity
            results.append((movie_data, similarity))
        
        return results
    
    def hybrid_search(self, query: str, target_emotions: Optional[Dict[str, float]] = None,
                     semantic_weight: float = 0.7, emotion_weight: float = 0.3,
                     top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        混合搜索：结合语义和情感相似度（修正版）
        
        参数:
            query: 查询文本
            target_emotions: 目标情感向量，如果为None则从查询中提取
            semantic_weight: 语义相似度权重
            emotion_weight: 情感相似度权重
            top_k: 返回结果数量
            
        返回:
            List[Tuple[Dict, float]]: 电影和综合分数列表
        """
        print(f"执行混合搜索（语义权重: {semantic_weight}, 情感权重: {emotion_weight})...")
        
        # 1. 获取目标情感向量
        if target_emotions is None:
            target_emotions = self.extract_emotions_from_query(query)
            print(f"从查询中提取的情感: {target_emotions}")
        
        # 2. 分别计算语义和情感相似度
        semantic_results = self.semantic_search(query, top_k=len(self.movies))
        emotion_results = self.emotion_search(target_emotions, top_k=len(self.movies))
        
        # 3. 构建分数字典
        semantic_scores = {}
        for movie, score in semantic_results:
            movie_id = movie['id']
            semantic_scores[movie_id] = score
        
        emotion_scores = {}
        for movie, score in emotion_results:
            movie_id = movie['id']
            emotion_scores[movie_id] = score
        
        # 4. 计算综合分数
        combined_scores = {}
        for movie in self.movies:
            movie_id = movie['id']
            semantic_score = semantic_scores.get(movie_id, 0)
            emotion_score = emotion_scores.get(movie_id, 0)
            
            # 加权综合分数
            combined_score = (semantic_score * semantic_weight) + (emotion_score * emotion_weight)
            combined_scores[movie_id] = combined_score
        
        # 5. 排序并返回结果
        sorted_movie_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for movie_id, combined_score in sorted_movie_ids:
            movie = next((m for m in self.movies if m['id'] == movie_id), None)
            if movie:
                # 获取单独的语义和情感分数用于显示
                sem_score = semantic_scores.get(movie_id, 0)
                emo_score = emotion_scores.get(movie_id, 0)
                
                movie_with_scores = movie.copy()
                movie_with_scores['semantic_score'] = sem_score
                movie_with_scores['emotion_score'] = emo_score
                
                results.append((movie_with_scores, combined_score))
        
        return results
    
    def extract_emotions_from_query(self, query: str) -> Dict[str, float]:
        """
        从查询文本中提取情感（修正版）
        
        参数:
            query: 查询文本
            
        返回:
            Dict[str, float]: 提取到的情感及其强度（英文标签）
        """
        # 简单的情感关键词映射（中文 -> 英文）
        emotion_keywords = {
            'joy': ['快乐', '开心', '高兴', '愉快', '欢乐', '喜悦', '搞笑', '幽默', '喜剧'],
            'sadness': ['悲伤', '难过', '伤心', '忧郁', '哀伤', '悲痛', '悲剧', '伤感'],
            'anger': ['愤怒', '生气', '气愤', '怒火', '愤慨', '恼怒', '暴力'],
            'fear': ['恐惧', '害怕', '恐怖', '惊吓', '惊悚', '恐慌', '可怕'],
            'love': ['爱', '爱情', '恋爱', '浪漫', '甜蜜', '温馨', '感人', '温暖'],
            'hope': ['希望', '期望', '盼望', '期待', '憧憬', '向往'],
            'loneliness': ['孤独', '孤单', '寂寞', '孤立', '独处', '疏离'],
            'inspiration': ['励志', '鼓舞', '激励', '振奋', '奋发', '向上'],
            'tension': ['紧张', '刺激', '悬疑', '惊险', '惊心动魄', '扣人心弦'],
            'peace': ['平静', '安宁', '宁静', '祥和', '安逸', '恬静']
        }
        
        # 初始化情感计数器
        emotion_counts = {emotion: 0 for emotion in emotion_keywords.keys()}
        
        # 检查查询中的情感关键词
        query_lower = query.lower()
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in query or keyword in query_lower:
                    emotion_counts[emotion] += 1
        
        # 计算情感强度（基于出现次数）
        extracted_emotions = {}
        total_hits = sum(emotion_counts.values())
        
        if total_hits > 0:
            for emotion, count in emotion_counts.items():
                if count > 0:
                    # 归一化到0-1范围
                    intensity = count / total_hits
                    extracted_emotions[emotion] = intensity
        else:
            # 如果没有检测到关键词，尝试基于查询整体情感
            if any(word in query for word in ['孤独', '寂寞', '无聊']):
                extracted_emotions = {'loneliness': 0.8, 'sadness': 0.2}
            elif any(word in query for word in ['开心', '快乐', '高兴']):
                extracted_emotions = {'joy': 0.8, 'love': 0.2}
            elif any(word in query for word in ['悲伤', '难过', '伤心']):
                extracted_emotions = {'sadness': 0.8, 'loneliness': 0.2}
            elif any(word in query for word in ['紧张', '刺激', '惊悚']):
                extracted_emotions = {'tension': 0.8, 'fear': 0.2}
            elif any(word in query for word in ['爱情', '浪漫', '甜蜜']):
                extracted_emotions = {'love': 0.8, 'joy': 0.2}
            else:
                # 默认返回一个通用情感分布
                extracted_emotions = {'joy': 0.3, 'hope': 0.3, 'inspiration': 0.4}
        
        print(f"查询分析结果: {extracted_emotions}")
        return extracted_emotions
    
    def visualize_emotion_profile(self, movie: Dict):
        """
        可视化电影情感分布
        
        参数:
            movie: 电影数据
        """
        emotion_profile = movie.get('emotion_profile', {})
        if not emotion_profile:
            print("该电影没有情感分布数据")
            return
        
        # 准备数据
        emotions = list(emotion_profile.keys())
        values = list(emotion_profile.values())
        
        # 只显示非零情感
        non_zero_data = [(e, v) for e, v in zip(emotions, values) if v > 0]
        if not non_zero_data:
            print("该电影的所有情感值都为0")
            return
        
        emotions, values = zip(*non_zero_data)
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
        bars = plt.barh(emotions, values, color=colors)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=10)
        
        plt.xlabel('情感强度')
        plt.title(f'{movie["title"]} 情感分布')
        plt.xlim(0, max(values) * 1.2)
        plt.tight_layout()
        
        # 显示图形
        plt.show()
    
    def get_recommendation_by_mood(self, mood_description: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        根据心情描述推荐电影
        
        参数:
            mood_description: 心情描述文本
            top_k: 返回结果数量
            
        返回:
            List[Tuple[Dict, float]]: 推荐的电影和分数
        """
        print(f"根据心情推荐: \"{mood_description}\"")
        
        # 从描述中提取情感
        target_emotions = self.extract_emotions_from_query(mood_description)
        
        if target_emotions:
            print(f"检测到情感: {target_emotions}")
            # 使用情感搜索
            return self.emotion_search(target_emotions, top_k)
        else:
            print("未检测到特定情感，使用语义搜索...")
            # 退回到语义搜索
            return self.semantic_search(mood_description, top_k)


def print_movie_results(query: str, results: List[Tuple[Dict, float]], show_emotions: bool = True):
    """
    打印电影搜索结果（修正版）
    
    参数:
        query: 查询文本
        results: 搜索结果列表
        show_emotions: 是否显示情感信息
    """
    print(f"\n🎬 查询: \"{query}\"")
    print("=" * 80)
    
    if not results:
        print("没有找到相关的电影")
        return
    
    for i, (movie, score) in enumerate(results, 1):
        # 创建相似度进度条
        bar_length = int(score * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        print(f"\n{i}. 🎥 {movie['title']} (综合匹配度: {score:.4f})")
        print(f"   📊 [{bar}]")
        
        # 显示语义和情感分数（如果存在）
        if 'semantic_score' in movie and 'emotion_score' in movie:
            sem_score = movie.get('semantic_score', 0)
            emo_score = movie.get('emotion_score', 0)
            if sem_score > 0 or emo_score > 0:
                print(f"   🔤 语义相似度: {sem_score:.4f}")
                print(f"   ❤️  情感相似度: {emo_score:.4f}")
        
        print(f"   📅 年份: {movie.get('year', '未知')}")
        
        genres = movie.get('genres', [])
        if isinstance(genres, list) and genres:
            print(f"   🎭 类型: {', '.join(genres)}")
        elif genres:
            print(f"   🎭 类型: {genres}")
        
        if show_emotions:
            mood_tags = movie.get('mood_tags', [])
            if mood_tags:
                if isinstance(mood_tags, list):
                    print(f"   🏷️  情感标签: {', '.join(mood_tags[:5])}")
                else:
                    print(f"   🏷️  情感标签: {mood_tags}")
            
            dominant_emotions = movie.get('dominant_emotions', [])
            if dominant_emotions:
                if isinstance(dominant_emotions, list):
                    print(f"   💫 主导情感: {', '.join(dominant_emotions[:3])}")
                else:
                    print(f"   💫 主导情感: {dominant_emotions}")
            
            # 显示情感向量摘要
            emotion_profile = movie.get('emotion_profile', {})
            if emotion_profile and isinstance(emotion_profile, dict):
                top_emotions = sorted([(k, v) for k, v in emotion_profile.items() if v > 0], 
                                     key=lambda x: x[1], reverse=True)[:3]
                if top_emotions:
                    emotion_str = ", ".join([f"{e}:{v:.2f}" for e, v in top_emotions])
                    print(f"   📈 主要情感: {emotion_str}")
        
        # 显示简要剧情
        plot = movie.get('plot', '')
        if plot:
            if len(plot) > 100:
                plot = plot[:97] + "..."
            print(f"   📖 简介: {plot}")


def main():
    """主函数：运行情感电影推荐系统"""
    print("=" * 80)
    print("🎭 情感氛围电影推荐系统 - 基于语义与情感分析")
    print("=" * 80)
    
    # 1. 创建推荐系统
    print("\n[1/4] 初始化推荐系统...")
    config.print_model_info()
    recommender = EmotionMovieRecommender()
    
    # 2. 加载电影数据
    print("\n[2/4] 加载电影数据...")
    
    # 尝试加载JSON格式的情感语料库
    json_path = "top_250_movies/top_rated_movie_emotions_20251202_214450.json"  # 修改为您的文件路径
    movies = recommender.load_movies_from_json(json_path)
    
    # 如果JSON加载失败，尝试CSV
    if not movies:
        csv_path = "top_250_movies/top_rated_movies_20251202_214450.csv"  # 修改为您的文件路径
        movies = recommender.load_movies_from_csv(csv_path)
    
    # 如果都没有，使用示例数据
    if not movies:
        movies = recommender.load_sample_movies()
    
    print(f"加载了 {len(movies)} 部电影，其中 {sum(1 for m in movies if m.get('emotion_profile'))} 部包含情感分析")
    
    # 3. 索引电影
    print("\n[3/4] 建立电影索引...")
    recommender.index_movies(movies)
    
    # 4. 演示不同搜索模式
    print("\n[4/4] 演示推荐功能")
    print("=" * 80)
    
    # 演示1: 纯语义搜索
    print("\n【演示 1: 语义搜索】")
    query1 = "希望与救赎的故事"
    results1 = recommender.semantic_search(query1, top_k=3)
    print_movie_results(query1, results1, show_emotions=False)
    
    # 演示2: 情感搜索
    print("\n\n【演示 2: 情感搜索】")
    target_emotions = {"hope": 0.4, "inspiration": 0.3, "sadness": 0.3}
    results2 = recommender.emotion_search(target_emotions, top_k=3)
    print_movie_results("希望与坚持的情感", results2)
    
    # 演示3: 混合搜索
    print("\n\n【演示 3: 混合搜索】")
    query3 = "既感人又充满希望的励志故事"
    results3 = recommender.hybrid_search(query3, top_k=3)
    print_movie_results(query3, results3)
    
    # 演示4: 根据心情推荐
    print("\n\n【演示 4: 根据心情推荐】")
    mood_query = "今天我感到有些孤独，但还抱有一丝希望"
    results4 = recommender.get_recommendation_by_mood(mood_query, top_k=3)
    print_movie_results(mood_query, results4)
    
    # 演示5: 可视化情感分布
    print("\n\n【演示 5: 情感分布可视化】")
    if movies and movies[0].get('emotion_profile'):
        print(f"显示电影《{movies[0]['title']}》的情感分布...")
        recommender.visualize_emotion_profile(movies[0])
    
    # 交互式搜索
    print("\n" + "=" * 80)
    print("🎯 交互式搜索 (输入 'quit' 或 'exit' 退出)")
    print("=" * 80)
    print("💡 模式说明:")
    print("  1. 语义搜索: 基于电影内容的文本匹配")
    print("  2. 情感搜索: 基于情感向量的相似度")
    print("  3. 混合搜索: 结合语义和情感（推荐）")
    print("  4. 心情推荐: 根据心情描述智能推荐")
    print("\n💡 示例查询:")
    print("  - '让人感动的电影' (语义搜索)")
    print("  - '孤独但充满希望' (情感搜索)")
    print("  - '紧张刺激的科幻片' (混合搜索)")
    print("  - '今天心情不好想看点温暖的' (心情推荐)")
    
    while True:
        try:
            print("\n" + "=" * 80)
            mode = input("\n请选择搜索模式 (1:语义, 2:情感, 3:混合, 4:心情, quit:退出): ").strip()
            
            if mode.lower() in ['quit', 'exit', 'q']:
                print("感谢使用！再见！")
                break
            
            if mode == '1':
                # 语义搜索
                query = input("请输入搜索关键词: ").strip()
                if not query:
                    print("查询不能为空")
                    continue
                
                top_k = input("返回结果数量 (默认5): ").strip()
                top_k = int(top_k) if top_k.isdigit() else 5
                
                results = recommender.semantic_search(query, top_k)
                print_movie_results(query, results, show_emotions=False)
                
            elif mode == '2':
                # 情感搜索
                print("请输入情感向量 (格式: 情感1:强度1, 情感2:强度2, ...)")
                print("可用情感:", ", ".join(recommender.fixed_emotion_labels))
                print("示例: joy:0.5, sadness:0.3, hope:0.2")
                
                emotion_input = input("情感向量: ").strip()
                if not emotion_input:
                    print("情感向量不能为空")
                    continue
                
                # 解析情感向量
                target_emotions = {}
                try:
                    for pair in emotion_input.split(','):
                        pair = pair.strip()
                        if ':' in pair:
                            emotion, value = pair.split(':', 1)
                            emotion = emotion.strip()
                            value = float(value.strip())
                            # 确保值在0-1之间
                            value = max(0.0, min(1.0, value))
                            target_emotions[emotion] = value
                except Exception as e:
                    print(f"格式错误: {e}")
                    print("请使用'情感:强度'格式，例如: joy:0.5, sadness:0.3")
                    continue
                
                top_k = input("返回结果数量 (默认5): ").strip()
                top_k = int(top_k) if top_k.isdigit() else 5
                
                results = recommender.emotion_search(target_emotions, top_k)
                print_movie_results(f"情感向量: {target_emotions}", results)
                
            elif mode == '3':
                # 混合搜索
                query = input("请输入搜索关键词: ").strip()
                if not query:
                    print("查询不能为空")
                    continue
                
                # 可选: 手动指定情感权重
                semantic_weight = input("语义权重 (默认0.7): ").strip()
                semantic_weight = float(semantic_weight) if semantic_weight else 0.7
                
                emotion_weight = input("情感权重 (默认0.3): ").strip()
                emotion_weight = float(emotion_weight) if emotion_weight else 0.3
                
                # 确保权重和为1
                if semantic_weight + emotion_weight != 1.0:
                    print("权重总和必须为1，已自动调整")
                    total = semantic_weight + emotion_weight
                    semantic_weight = semantic_weight / total
                    emotion_weight = emotion_weight / total
                
                top_k = input("返回结果数量 (默认5): ").strip()
                top_k = int(top_k) if top_k.isdigit() else 5
                
                results = recommender.hybrid_search(query, semantic_weight=semantic_weight, 
                                                  emotion_weight=emotion_weight, top_k=top_k)
                print_movie_results(query, results)
                
            elif mode == '4':
                # 心情推荐
                mood = input("请描述你现在的心情: ").strip()
                if not mood:
                    print("心情描述不能为空")
                    continue
                
                top_k = input("返回结果数量 (默认5): ").strip()
                top_k = int(top_k) if top_k.isdigit() else 5
                
                results = recommender.get_recommendation_by_mood(mood, top_k)
                print_movie_results(f"心情: {mood}", results)
                
            else:
                print("无效的模式选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n程序被中断，感谢使用！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ 情感氛围电影推荐系统演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    # 运行主程序
    main()
