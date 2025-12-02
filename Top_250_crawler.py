#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMDB影史评分前250电影情感语料库爬虫程序
基于Top Rated接口，获取评分最高的电影
"""

import requests
import json
import time
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional


class TMDBTopRatedCrawler:
    """TMDB Top Rated电影数据爬虫"""
    
    def __init__(self, api_key: str):
        """
        初始化爬虫
        
        参数:
            api_key: TMDB API密钥
        """
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        
        # 情感词典定义（用于情感分析）
        self.emotion_lexicon = {
            # 基本情感
            'joy': ['happy', 'joy', 'fun', 'funny', 'laughter', 'smile', 'cheerful', 
                   'delight', 'euphoria', 'bliss', 'elation', 'glee', '喜剧', '欢乐', '开心', '愉快'],
            'sadness': ['sad', 'sadness', 'grief', 'sorrow', 'melancholy', 'depression',
                       'tear', 'cry', 'mourn', 'heartbreak', 'despair', 'misery', 
                       '悲剧', '悲伤', '难过', '忧郁'],
            'anger': ['anger', 'angry', 'rage', 'fury', 'wrath', 'outrage', 'frustration',
                     'resentment', 'hostility', 'irritation', 'annoyance', '愤怒', '生气', '怒火'],
            'fear': ['fear', 'scary', 'terror', 'horror', 'dread', 'panic', 'anxiety',
                    'fright', 'apprehension', 'trepidation', 'phobia', '恐惧', '恐怖', '害怕'],
            'love': ['love', 'romance', 'passion', 'affection', 'adore', 'cherish',
                    'devotion', 'intimacy', 'tenderness', 'fondness', 'infatuation',
                    '爱情', '浪漫', '温馨', '甜蜜'],
            'hope': ['hope', 'hopeful', 'optimism', 'faith', 'confidence', 'expectation',
                    'aspiration', 'dream', 'wish', 'anticipation', '希望', '梦想', '期待'],
            'loneliness': ['lonely', 'loneliness', 'isolated', 'solitude', 'alone',
                          'abandoned', 'desolate', 'secluded', 'forsaken', '孤独', '孤单', '寂寞'],
            'inspiration': ['inspire', 'inspiring', 'motivation', 'encouraging', 
                           'uplifting', 'empowering', 'moving', 'touching', '励志', '鼓舞', '激励'],
            'tension': ['tense', 'tension', 'suspense', 'thrilling', 'nerve-racking',
                       'nail-biting', 'edge-of-seat', 'anxious', 'stressful', '紧张', '悬疑', '惊悚'],
            'peace': ['peace', 'peaceful', 'calm', 'serene', 'tranquil', 'relaxed',
                     'quiet', 'soothing', 'placid', 'composed', '平静', '安宁', '宁静']
        }
        
    def get_top_rated_movies(self, page: int = 1, language: str = 'zh-CN') -> List[Dict]:
        """
        获取Top Rated电影列表
        
        参数:
            page: 页码
            language: 语言
            
        返回:
            List[Dict]: 电影列表
        """
        url = f"{self.base_url}/movie/top_rated"
        params = {
            'api_key': self.api_key,
            'language': language,
            'page': page
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            movies = []
            for movie in data.get('results', []):
                movie_info = {
                    'id': movie.get('id'),
                    'title': movie.get('title', ''),
                    'original_title': movie.get('original_title', ''),
                    'overview': movie.get('overview', ''),
                    'release_date': movie.get('release_date', ''),
                    'vote_average': movie.get('vote_average', 0),
                    'vote_count': movie.get('vote_count', 0),
                    'popularity': movie.get('popularity', 0),
                    'poster_path': movie.get('poster_path', ''),
                    'top_rated_rank': (page - 1) * 20 + data.get('results', []).index(movie) + 1  # 在Top Rated列表中的排名
                }
                movies.append(movie_info)
            
            return movies
            
        except Exception as e:
            print(f"获取Top Rated电影列表失败 (页码 {page}): {e}")
            return []
    
    def get_movie_details(self, movie_id: int, language: str = 'zh-CN') -> Dict:
        """
        获取电影详细信息
        
        参数:
            movie_id: 电影ID
            language: 语言
            
        返回:
            Dict: 电影详细信息
        """
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            'api_key': self.api_key,
            'language': language,
            'append_to_response': 'credits,keywords'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # 提取导演信息
            director = ''
            for person in data.get('credits', {}).get('crew', []):
                if person.get('job') == 'Director':
                    director = person.get('name', '')
                    break
            
            # 提取主要演员
            cast = []
            for person in data.get('credits', {}).get('cast', [])[:5]:
                cast.append(person.get('name', ''))
            
            # 提取关键词
            keywords = [kw['name'] for kw in data.get('keywords', {}).get('keywords', [])]
            
            details = {
                'genres': [genre['name'] for genre in data.get('genres', [])],
                'runtime': data.get('runtime', 0),
                'budget': data.get('budget', 0),
                'revenue': data.get('revenue', 0),
                'director': director,
                'cast': cast,
                'keywords': keywords,
                'tagline': data.get('tagline', ''),
                'status': data.get('status', ''),
                'imdb_id': data.get('imdb_id', ''),
                'production_companies': [company['name'] for company in data.get('production_companies', [])],
                'production_countries': [country['name'] for country in data.get('production_countries', [])]
            }
            
            return details
            
        except Exception as e:
            print(f"获取电影{movie_id}详情失败: {e}")
            return {}
    
    def get_movie_reviews(self, movie_id: int, language: str = 'en-US', max_reviews: int = 10) -> List[Dict]:
        """
        获取电影影评
        
        参数:
            movie_id: 电影ID
            language: 语言
            max_reviews: 最大影评数
            
        返回:
            List[Dict]: 影评列表
        """
        url = f"{self.base_url}/movie/{movie_id}/reviews"
        params = {
            'api_key': self.api_key,
            'language': language,
            'page': 1
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            reviews = []
            for review in data.get('results', [])[:max_reviews]:
                review_data = {
                    'author': review.get('author', 'Anonymous'),
                    'content': review.get('content', ''),
                    'created_at': review.get('created_at', ''),
                    'rating': None,  # TMDB评论没有评分
                    'url': f"https://www.themoviedb.org/review/{review.get('id', '')}",
                    'source': 'tmdb'
                }
                
                # 简单情感分析
                sentiment = self.analyze_review_sentiment(review_data['content'])
                review_data['sentiment'] = sentiment
                
                reviews.append(review_data)
            
            return reviews
            
        except Exception as e:
            print(f"获取电影{movie_id}影评失败: {e}")
            return []
    
    def analyze_review_sentiment(self, text: str) -> Dict:
        """
        简单影评情感分析
        
        参数:
            text: 影评文本
            
        返回:
            Dict: 情感分析结果
        """
        if not text:
            return {'sentiment': 'neutral', 'score': 0.5}
        
        text_lower = text.lower()
        
        # 简单的关键词匹配
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'love', 'like', 'enjoy', 'best', 'awesome', '推荐', '精彩', '经典']
        negative_words = ['bad', 'terrible', 'awful', 'poor', 'disappointing', 
                         'hate', 'dislike', 'worst', 'boring', '糟糕', '失望', '无聊']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return {'sentiment': 'neutral', 'score': 0.5}
        
        sentiment_score = positive_count / total
        
        if sentiment_score > 0.6:
            sentiment = 'positive'
        elif sentiment_score < 0.4:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {'sentiment': sentiment, 'score': sentiment_score}
    
    def analyze_movie_emotion(self, overview: str, tagline: str = '', keywords: List[str] = [], genres: List[str] = None) -> Dict:
        """
        分析电影情感氛围（增强版）
        
        参数:
            overview: 电影简介
            tagline: 宣传语
            keywords: 关键词
            genres: 电影类型（用于后备情感分析）
            
        返回:
            Dict: 情感分析结果
        """
        # 1. 主分析：合并所有文本进行分析
        combined_text = f"{tagline} {overview} {' '.join(keywords)}".lower()
        
        emotion_scores = {}
        total_score = 0
        
        for emotion, emotion_words in self.emotion_lexicon.items():
            score = 0
            for word in emotion_words:
                # 改进的匹配：包含单词边界检查和模糊匹配
                if len(word) <= 3:  # 短词
                    pattern = r'\b' + re.escape(word) + r'\b'
                    matches = re.findall(pattern, combined_text)
                    score += len(matches)
                else:  # 长词
                    score += combined_text.count(word) * (2 if len(word) > 4 else 1)
            
            if score > 0:
                emotion_scores[emotion] = score
                total_score += score
        
        # 2. 如果主分析没有结果，使用后备策略
        if not emotion_scores:
            emotion_scores = self.fallback_emotion_analysis(overview, tagline, keywords, genres)
            total_score = sum(emotion_scores.values())
        
        # 3. 归一化情感分数
        normalized_scores = {}
        if total_score > 0:
            for emotion, score in emotion_scores.items():
                normalized_scores[emotion] = round(score / total_score, 3)
        
        # 4. 获取主导情感
        dominant_emotions = []
        if normalized_scores:
            sorted_emotions = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
            dominant_emotions = [emotion for emotion, score in sorted_emotions[:3] if score > 0.1]
        
        # 5. 生成情绪标签
        mood_tags = self.generate_mood_tags(normalized_scores)
        
        return {
            'emotion_profile': normalized_scores,
            'dominant_emotions': dominant_emotions,
            'mood_tags': mood_tags,
            'emotional_complexity': len(normalized_scores)
        }
    
    def fallback_emotion_analysis(self, overview: str, tagline: str, keywords: List[str], genres: List[str] = None) -> Dict[str, float]:
        """
        后备情感分析策略
        当主分析失败时使用
        """
        emotion_scores = {}
        
        # 策略1：基于电影标题分析
        title_text = f"{tagline} {overview}".lower()
        
        # 扩展的情感词典（更全面的词汇）
        extended_emotion_lexicon = {
            'joy': ['happy', 'joy', 'fun', 'funny', 'laughter', 'smile', 'cheerful', 
                   'delight', 'euphoria', 'bliss', 'elation', 'glee', 'comic', 'humor',
                   'lighthearted', '喜剧', '欢乐', '开心', '愉快', '搞笑', '幽默'],
            'sadness': ['sad', 'sadness', 'grief', 'sorrow', 'melancholy', 'depression',
                       'tear', 'cry', 'mourn', 'heartbreak', 'despair', 'misery', 'tragedy',
                       'loss', 'death', 'dying', 'grave', 'funeral', '悲剧', '悲伤', '难过'],
            'anger': ['anger', 'angry', 'rage', 'fury', 'wrath', 'outrage', 'frustration',
                     'resentment', 'hostility', 'irritation', 'annoyance', 'violence',
                     'fight', 'war', 'conflict', 'battle', '愤怒', '生气', '怒火', '暴力'],
            'fear': ['fear', 'scary', 'terror', 'horror', 'dread', 'panic', 'anxiety',
                    'fright', 'apprehension', 'trepidation', 'phobia', 'monster',
                    'ghost', 'haunted', 'supernatural', '恐惧', '恐怖', '害怕', '惊吓'],
            'love': ['love', 'romance', 'passion', 'affection', 'adore', 'cherish',
                    'devotion', 'intimacy', 'tenderness', 'fondness', 'infatuation',
                    'relationship', 'couple', 'marriage', 'wedding', '爱情', '浪漫', '温馨'],
            'hope': ['hope', 'hopeful', 'optimism', 'faith', 'confidence', 'expectation',
                    'aspiration', 'dream', 'wish', 'anticipation', 'future', 'better',
                    'improve', 'recover', 'heal', '希望', '梦想', '期待', '信念'],
            'loneliness': ['lonely', 'loneliness', 'isolated', 'solitude', 'alone',
                          'abandoned', 'desolate', 'secluded', 'forsaken', '孤独', '孤单'],
            'tension': ['tense', 'tension', 'suspense', 'thrilling', 'nerve-racking',
                       'nail-biting', 'edge-of-seat', 'anxious', 'stressful', '紧张', '悬疑'],
            'peace': ['peace', 'peaceful', 'calm', 'serene', 'tranquil', 'relaxed',
                     'quiet', 'soothing', 'placid', 'composed', '平静', '安宁', '宁静'],
            'inspiration': ['inspire', 'inspiring', 'motivation', 'encouraging', 
                           'uplifting', 'empowering', 'moving', 'touching', '励志', '鼓舞']
        }
        
        # 使用扩展词典分析
        for emotion, words in extended_emotion_lexicon.items():
            score = 0
            for word in words:
                if word in title_text:
                    score += 2 if len(word) > 4 else 1
            if score > 0:
                emotion_scores[emotion] = score
        
        # 策略2：基于电影类型推断情感
        if genres and not emotion_scores:
            genre_emotion_map = {
                '喜剧': 'joy', '喜剧片': 'joy', 'Comedy': 'joy',
                '剧情': 'sadness', '剧情片': 'sadness', 'Drama': 'sadness',
                '恐怖': 'fear', '恐怖片': 'fear', 'Horror': 'fear',
                '爱情': 'love', '爱情片': 'love', 'Romance': 'love',
                '科幻': 'hope', '科幻片': 'hope', 'Science Fiction': 'hope',
                '惊悚': 'tension', '惊悚片': 'tension', 'Thriller': 'tension',
                '动作': 'tension', '动作片': 'tension', 'Action': 'tension',
                '冒险': 'joy', '冒险片': 'joy', 'Adventure': 'joy',
                '动画': 'joy', '动画片': 'joy', 'Animation': 'joy',
                '家庭': 'joy', '家庭片': 'joy', 'Family': 'joy',
                '战争': 'fear', '战争片': 'fear', 'War': 'fear',
                '犯罪': 'anger', '犯罪片': 'anger', 'Crime': 'anger',
                '悬疑': 'tension', '悬疑片': 'tension', 'Mystery': 'tension'
            }
            
            for genre in genres:
                emotion = genre_emotion_map.get(genre)
                if emotion:
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0) + 2
        
        # 策略3：基于已知电影信息（硬编码一些知名电影的情感）
        known_movie_emotions = {
            '教父': {'tension': 3, 'anger': 2, 'sadness': 2},
            '教父2': {'tension': 3, 'anger': 2, 'sadness': 3},
            '辛德勒的名单': {'sadness': 4, 'hope': 2, 'inspiration': 3},
            '肖申克的救赎': {'hope': 4, 'sadness': 2, 'inspiration': 3},
            '盗梦空间': {'tension': 3, 'hope': 2, 'fear': 1},
            '阿甘正传': {'hope': 3, 'joy': 2, 'inspiration': 3},
            '泰坦尼克号': {'love': 4, 'sadness': 3, 'fear': 2},
            '美丽人生': {'hope': 3, 'joy': 2, 'sadness': 3},
            '钢琴家': {'sadness': 4, 'fear': 3, 'hope': 2},
            '拯救大兵瑞恩': {'fear': 3, 'anger': 2, 'hope': 2},
            '指环王': {'hope': 3, 'joy': 2, 'tension': 2},
            '哈利波特': {'joy': 3, 'fear': 2, 'hope': 2},
            '星球大战': {'hope': 3, 'joy': 2, 'tension': 2},
            '黑客帝国': {'tension': 3, 'hope': 2, 'fear': 1},
            '沉默的羔羊': {'fear': 4, 'tension': 3, 'anger': 1},
            '低俗小说': {'joy': 3, 'tension': 2, 'anger': 1},
            '飞越疯人院': {'hope': 3, 'sadness': 2, 'anger': 2},
            '闪灵': {'fear': 4, 'tension': 3, 'anger': 1},
            '公民凯恩': {'sadness': 3, 'anger': 2, 'hope': 1},
            '七武士': {'tension': 3, 'hope': 2, 'sadness': 2}
        }
        
        # 检查是否是已知电影
        title_lower = overview.lower() if overview else ''
        for known_title, emotions in known_movie_emotions.items():
            if known_title in title_lower:
                for emotion, score in emotions.items():
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score
                break
        
        # 策略4：如果还是没有结果，给一个默认的情感分布
        if not emotion_scores:
            # 根据文本长度和内容猜测
            text_length = len(overview) + len(tagline)
            if text_length < 50:  # 文本很短
                # 给一个中性偏积极的情感分布
                emotion_scores = {'hope': 2, 'inspiration': 1, 'joy': 1}
            else:
                # 根据常见词汇猜测
                if any(word in title_text for word in ['war', 'battle', 'fight', '战争', '战斗']):
                    emotion_scores = {'fear': 3, 'tension': 2, 'anger': 1}
                elif any(word in title_text for word in ['love', 'romance', '爱', '爱情']):
                    emotion_scores = {'love': 4, 'joy': 2, 'hope': 1}
                elif any(word in title_text for word in ['death', 'die', 'dead', '死亡', '死去']):
                    emotion_scores = {'sadness': 4, 'hope': 1, 'inspiration': 1}
                else:
                    # 通用情感分布
                    emotion_scores = {'hope': 2, 'inspiration': 2, 'joy': 1, 'sadness': 1}
        
        return emotion_scores

    
    def generate_mood_tags(self, emotion_profile: Dict) -> List[str]:
        """
        根据情感分布生成情绪标签（增强版）
        
        参数:
            emotion_profile: 情感分布
            
        返回:
            List[str]: 情绪标签列表
        """
        if not emotion_profile:
            # 如果没有情感分析结果，返回通用标签
            return ['情感丰富', '引人深思', '值得一看']
        
        tags = []
        
        # 情感强度阈值（降低阈值，更容易生成标签）
        strong_threshold = 0.15  
        moderate_threshold = 0.05
        
        # 基础情感标签
        for emotion, score in emotion_profile.items():
            if score >= strong_threshold:
                tags.append(f"非常{emotion}")
            elif score >= moderate_threshold:
                tags.append(f"有些{emotion}")
        
        # 特殊组合标签
        if 'joy' in emotion_profile and 'love' in emotion_profile:
            if emotion_profile.get('joy', 0) > 0.1 and emotion_profile.get('love', 0) > 0.1:
                tags.append("温暖治愈")
        
        if 'sadness' in emotion_profile and 'hope' in emotion_profile:
            if emotion_profile.get('sadness', 0) > 0.1 and emotion_profile.get('hope', 0) > 0.05:
                tags.append("悲伤但充满希望")
        
        if 'fear' in emotion_profile and emotion_profile.get('fear', 0) > 0.15:
            tags.append("紧张刺激")
        
        if 'peace' in emotion_profile and emotion_profile.get('peace', 0) > 0.1:
            tags.append("心灵平静")
        
        if 'inspiration' in emotion_profile and emotion_profile.get('inspiration', 0) > 0.1:
            tags.append("励志感人")
        
        # 如果标签太少，添加一些通用标签
        if len(tags) < 2:
            # 根据最高分的情感添加通用标签
            if emotion_profile:
                max_emotion = max(emotion_profile.items(), key=lambda x: x[1])[0]
                if max_emotion in ['joy', 'love']:
                    tags.append("情感丰富")
                    tags.append("值得推荐")
                elif max_emotion in ['sadness', 'fear', 'tension']:
                    tags.append("引人深思")
                    tags.append("情感强烈")
                else:
                    tags.append("情感真挚")
                    tags.append("值得一看")
        
        return list(set(tags))[:6]  # 最多6个标签

    
    def crawl_top_rated_movies(self, num_movies: int = 250, max_reviews_per_movie: int = 5) -> List[Dict]:
        """
        爬取Top Rated电影
        
        参数:
            num_movies: 要爬取的电影数量
            max_reviews_per_movie: 每部电影最大影评数
            
        返回:
            List[Dict]: 完整的电影数据
        """
        print(f"开始爬取影史评分前{num_movies}部电影...")
        
        all_movies = []
        collected_ids = set()
        
        # 计算需要的页数（每页20部）
        pages_needed = (num_movies // 20) + 1
        if num_movies % 20 == 0:
            pages_needed = num_movies // 20
        
        # 收集电影列表
        movie_list = []
        for page in range(1, pages_needed + 1):
            print(f"获取第 {page} 页Top Rated电影列表...")
            movies = self.get_top_rated_movies(page=page)
            
            if not movies:
                print(f"第 {page} 页没有数据，停止获取")
                break
            
            for movie in movies:
                movie_id = movie['id']
                if movie_id not in collected_ids:
                    movie_list.append(movie)
                    collected_ids.add(movie_id)
            
            time.sleep(0.5)  # 礼貌延迟
            
            if len(movie_list) >= num_movies:
                break
        
        # 限制到指定数量
        movie_list = movie_list[:num_movies]
        
        # 处理每部电影
        for i, movie in enumerate(movie_list, 1):
            movie_id = movie['id']
            top_rated_rank = movie.get('top_rated_rank', i)
            
            print(f"\n[{i}/{len(movie_list)}] 处理电影: {movie['title']} (TMDB Top Rated排名: {top_rated_rank})")
            print(f"  评分: {movie['vote_average']}/10, 投票数: {movie['vote_count']:,}")
            
            # 获取详细信息
            print("  获取详细信息...")
            details = self.get_movie_details(movie_id)
            time.sleep(0.3)
            
            # 获取影评
            print("  获取影评...")
            reviews = self.get_movie_reviews(movie_id, max_reviews=max_reviews_per_movie)
            time.sleep(0.3)
            
            # 分析情感
            print("  分析情感氛围...")
            emotion_analysis = self.analyze_movie_emotion(
    overview=movie.get('overview', ''),
    tagline=details.get('tagline', ''),
    keywords=details.get('keywords', []),
    genres=details.get('genres', [])  # 添加这一行
)
            
            # 构建完整电影数据
            movie_data = {
                'id': movie_id,
                'title': movie.get('title', ''),
                'original_title': movie.get('original_title', ''),
                'release_date': movie.get('release_date', ''),
                'release_year': movie.get('release_date', '')[:4] if movie.get('release_date') else '',
                'overview': movie.get('overview', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'popularity': movie.get('popularity', 0),
                
                # Top Rated特定信息
                'tmdb_top_rated_rank': top_rated_rank,
                
                # 详细信息
                'genres': details.get('genres', []),
                'runtime': details.get('runtime', 0),
                'director': details.get('director', ''),
                'cast': details.get('cast', []),
                'tagline': details.get('tagline', ''),
                'keywords': details.get('keywords', []),
                'imdb_id': details.get('imdb_id', ''),
                
                # 情感分析结果
                'emotion_profile': emotion_analysis.get('emotion_profile', {}),
                'dominant_emotions': emotion_analysis.get('dominant_emotions', []),
                'mood_tags': emotion_analysis.get('mood_tags', []),
                'emotional_complexity': emotion_analysis.get('emotional_complexity', 0),
                
                # 影评
                'reviews': reviews,
                'review_count': len(reviews),
                
                # 爬虫信息
                'source': 'tmdb_top_rated',
                'crawl_date': datetime.now().isoformat()
            }
            
            all_movies.append(movie_data)
            
            print(f"  ✓ 完成: {movie['title']}")
            print(f"     情感标签: {', '.join(emotion_analysis.get('mood_tags', []))}")
            print(f"     主导情感: {', '.join(emotion_analysis.get('dominant_emotions', []))}")
            
            # 礼貌延迟
            time.sleep(0.5)
        
        return all_movies
    
    def save_data(self, movie_data: List[Dict], output_dir: str = 'top_rated_movies'):
        """
        保存爬取的数据
        
        参数:
            movie_data: 电影数据列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n开始保存数据到目录: {output_dir}")
        
        # 1. 保存完整的JSON格式情感语料库
        json_path = os.path.join(output_dir, f'top_rated_movie_emotions_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(movie_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 情感语料库已保存: {json_path} ({len(movie_data)} 部电影)")
        
        # 2. 保存为CSV格式（用于原有程序）
        csv_path = os.path.join(output_dir, f'top_rated_movies_{timestamp}.csv')
        self.save_as_csv(movie_data, csv_path)
        print(f"✓ CSV格式已保存: {csv_path}")
        
        # 3. 保存影评数据
        reviews_path = os.path.join(output_dir, f'top_rated_reviews_{timestamp}.json')
        self.save_reviews(movie_data, reviews_path)
        print(f"✓ 影评数据已保存: {reviews_path}")
        
        # 4. 保存统计信息
        stats_path = os.path.join(output_dir, f'top_rated_statistics_{timestamp}.txt')
        self.save_statistics(movie_data, stats_path)
        print(f"✓ 统计信息已保存: {stats_path}")
        
        # 5. 保存情感分析专用格式
        emotion_csv_path = os.path.join(output_dir, f'top_rated_emotion_vectors_{timestamp}.csv')
        self.save_emotion_vectors(movie_data, emotion_csv_path)
        print(f"✓ 情感向量已保存: {emotion_csv_path}")
        
        # 6. 保存增强版CSV（用于电影推荐程序）
        enhanced_csv_path = os.path.join(output_dir, f'enhanced_top_rated_movies_{timestamp}.csv')
        self.save_enhanced_csv(movie_data, enhanced_csv_path)
        print(f"✓ 增强版CSV已保存: {enhanced_csv_path}")
        
        # 7. 保存排名信息
        ranking_path = os.path.join(output_dir, f'top_rated_ranking_{timestamp}.csv')
        self.save_ranking(movie_data, ranking_path)
        print(f"✓ 排名信息已保存: {ranking_path}")
        
        # 返回文件路径供后续使用
        return {
            'json_corpus': json_path,
            'csv_data': csv_path,
            'enhanced_csv': enhanced_csv_path,
            'reviews': reviews_path,
            'emotion_vectors': emotion_csv_path,
            'ranking': ranking_path
        }
    
    def save_as_csv(self, movie_data: List[Dict], csv_path: str):
        """保存为CSV格式（兼容原有程序）"""
        rows = []
        for movie in movie_data:
            row = {
                'movie_id': movie['id'],
                'title': movie['title'],
                'original_title': movie.get('original_title', ''),
                'plot': movie.get('overview', ''),
                'genres': '|'.join(movie.get('genres', [])),
                'year': movie.get('release_year', ''),
                'rating': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'director': movie.get('director', ''),
                'runtime': movie.get('runtime', 0),
                'tagline': movie.get('tagline', ''),
                'mood_tags': '|'.join(movie.get('mood_tags', [])),
                'dominant_emotions': '|'.join(movie.get('dominant_emotions', [])),
                'review_count': movie.get('review_count', 0),
                'tmdb_top_rated_rank': movie.get('tmdb_top_rated_rank', 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    def save_enhanced_csv(self, movie_data: List[Dict], csv_path: str):
        """保存为增强版CSV格式（包含更多情感信息）"""
        rows = []
        for movie in movie_data:
            # 计算平均影评情感
            avg_sentiment = 0.5
            if movie.get('reviews'):
                sentiment_scores = [r.get('sentiment', {}).get('score', 0.5) for r in movie['reviews']]
                avg_sentiment = round(np.mean(sentiment_scores), 3)
            
            row = {
                'movie_id': movie['id'],
                'title': movie['title'],
                'original_title': movie.get('original_title', ''),
                'plot': movie.get('overview', ''),
                'tagline': movie.get('tagline', ''),
                'genres': '|'.join(movie.get('genres', [])),
                'year': movie.get('release_year', ''),
                'rating': movie.get('vote_average', 0),
                'runtime': movie.get('runtime', 0),
                'director': movie.get('director', ''),
                'main_cast': '|'.join(movie.get('cast', [])[:3]),
                
                # Top Rated信息
                'tmdb_top_rated_rank': movie.get('tmdb_top_rated_rank', 0),
                
                # 情感信息
                'mood_tags': '|'.join(movie.get('mood_tags', [])),
                'dominant_emotions': '|'.join(movie.get('dominant_emotions', [])),
                'emotional_complexity': movie.get('emotional_complexity', 0),
                
                # 影评信息
                'review_count': movie.get('review_count', 0),
                'avg_review_sentiment': avg_sentiment,
                
                # 其他信息
                'popularity': movie.get('popularity', 0),
                'vote_count': movie.get('vote_count', 0),
                
                # 情感向量（简化版，只取前3个）
                'emotion_vector': self.get_emotion_vector_string(movie.get('emotion_profile', {}))
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    def save_ranking(self, movie_data: List[Dict], ranking_path: str):
        """保存排名信息"""
        rows = []
        for movie in movie_data:
            row = {
                'rank': movie.get('tmdb_top_rated_rank', 0),
                'title': movie['title'],
                'original_title': movie.get('original_title', ''),
                'year': movie.get('release_year', ''),
                'rating': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'director': movie.get('director', ''),
                'genres': '|'.join(movie.get('genres', [])),
                'mood_tags': '|'.join(movie.get('mood_tags', [])[:3]),
                'imdb_id': movie.get('imdb_id', '')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('rank')
        df.to_csv(ranking_path, index=False, encoding='utf-8-sig')
    
    def get_emotion_vector_string(self, emotion_profile: Dict) -> str:
        """将情感向量转换为字符串"""
        if not emotion_profile:
            return ''
        
        # 取前3个最强烈的情感
        top_emotions = sorted(emotion_profile.items(), key=lambda x: x[1], reverse=True)[:3]
        return '|'.join([f"{emotion}:{score:.3f}" for emotion, score in top_emotions])
    
    def save_reviews(self, movie_data: List[Dict], reviews_path: str):
        """保存影评数据"""
        all_reviews = []
        for movie in movie_data:
            for review in movie.get('reviews', []):
                review_data = {
                    'movie_id': movie.get('id'),
                    'movie_title': movie.get('title', ''),
                    'rank': movie.get('tmdb_top_rated_rank', 0),
                    'author': review.get('author', ''),
                    'content': review.get('content', ''),
                    'sentiment': review.get('sentiment', {}).get('sentiment', ''),
                    'sentiment_score': review.get('sentiment', {}).get('score', 0),
                    'created_at': review.get('created_at', ''),
                    'source': review.get('source', '')
                }
                all_reviews.append(review_data)
        
        with open(reviews_path, 'w', encoding='utf-8') as f:
            json.dump(all_reviews, f, ensure_ascii=False, indent=2)
    
    def save_statistics(self, movie_data: List[Dict], stats_path: str):
        """保存统计信息"""
        total_movies = len(movie_data)
        total_reviews = sum(len(movie.get('reviews', [])) for movie in movie_data)
        
        # 情感统计
        emotion_counter = {}
        mood_tag_counter = {}
        
        for movie in movie_data:
            for emotion in movie.get('dominant_emotions', []):
                emotion_counter[emotion] = emotion_counter.get(emotion, 0) + 1
            
            for tag in movie.get('mood_tags', []):
                mood_tag_counter[tag] = mood_tag_counter.get(tag, 0) + 1
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"TMDB影史评分前{total_movies}电影统计信息\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"电影总数: {total_movies}\n")
            f.write(f"影评总数: {total_reviews}\n")
            f.write(f"平均每部电影影评数: {total_reviews/total_movies:.1f}\n")
            
            # 平均评分
            avg_rating = np.mean([m['vote_average'] for m in movie_data])
            avg_votes = np.mean([m['vote_count'] for m in movie_data])
            f.write(f"平均评分: {avg_rating:.2f}/10\n")
            f.write(f"平均投票数: {avg_votes:,.0f}\n\n")
            
            f.write("主导情感分布:\n")
            for emotion, count in sorted(emotion_counter.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_movies) * 100
                f.write(f"  {emotion}: {count} ({percentage:.1f}%)\n")
            
            f.write("\n情绪标签分布 (前20):\n")
            for tag, count in sorted(mood_tag_counter.items(), key=lambda x: x[1], reverse=True)[:20]:
                percentage = (count / total_movies) * 100
                f.write(f"  {tag}: {count} ({percentage:.1f}%)\n")
            
            f.write("\nTop 10电影:\n")
            sorted_movies = sorted(movie_data, key=lambda x: x.get('tmdb_top_rated_rank', 0))
            for movie in sorted_movies[:10]:
                f.write(f"\n  {movie.get('tmdb_top_rated_rank', 0)}. 《{movie['title']}》\n")
                f.write(f"     评分: {movie['vote_average']}/10, 投票: {movie['vote_count']:,}\n")
                f.write(f"     情感标签: {', '.join(movie.get('mood_tags', []))}\n")
                f.write(f"     主导情感: {', '.join(movie.get('dominant_emotions', []))}\n")
                f.write(f"     导演: {movie.get('director', '')}\n")
    
    def save_emotion_vectors(self, movie_data: List[Dict], csv_path: str):
        """保存情感向量（用于机器学习）"""
        # 获取所有情感维度
        all_emotions = set()
        for movie in movie_data:
            all_emotions.update(movie.get('emotion_profile', {}).keys())
        
        # 构建表头
        headers = ['movie_id', 'title', 'year', 'rank'] + sorted(list(all_emotions))
        
        # 构建数据行
        rows = []
        for movie in movie_data:
            emotion_profile = movie.get('emotion_profile', {})
            row = [
                movie['id'], 
                movie['title'], 
                movie.get('release_year', ''),
                movie.get('tmdb_top_rated_rank', 0)
            ]
            
            for emotion in sorted(list(all_emotions)):
                row.append(emotion_profile.get(emotion, 0.0))
            
            rows.append(row)
        
        # 保存为CSV
        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')


def main():
    """主函数"""
    print("=" * 80)
    print("🎬 TMDB影史评分前250电影情感语料库爬虫程序")
    print("=" * 80)
    print("功能:")
    print("  • 从TMDB Top Rated列表获取高评分电影")
    print("  • 自动情感分析，生成情感标签")
    print("  • 生成影史评分前250电影语料库")
    print("=" * 80)
    
    # 获取TMDB API密钥
    TMDB_API_KEY = "c095e562a2d5b49381ac1977284f8a04"
    
    # 获取TMDB API密钥
    tmdb_api_key = TMDB_API_KEY
    
    if not tmdb_api_key:
        print("错误: 请在代码中设置TMDB API密钥")
        print("获取方法: https://www.themoviedb.org/settings/api")
        return
    
    print(f"使用预设的TMDB API密钥: {tmdb_api_key[:8]}...")
    
    # 配置参数
    try:
        num_movies = int(input(f"\n要爬取多少部Top Rated电影? (默认250): ") or "250")
        output_dir = input(f"输出目录? (默认'top_250_movies'): ") or "top_250_movies"
    except:
        num_movies = 250
        output_dir = "top_250_movies"
    
    print(f"\n配置确认:")
    print(f"  • 电影数量: {num_movies} (影史评分前{num_movies}部)")
    print(f"  • 输出目录: {output_dir}")
    print(f"  • 情感分析: 开启")
    print(f"  • 预计耗时: {num_movies * 1.5 / 60:.1f} 分钟")
    
    confirm = input(f"\n开始爬取? (y/n): ").strip().lower()
    if confirm != 'y':
        print("程序取消")
        return
    
    # 创建爬虫实例
    try:
        crawler = TMDBTopRatedCrawler(tmdb_api_key)
        
        # 爬取Top Rated电影数据
        movie_data = crawler.crawl_top_rated_movies(num_movies=num_movies, max_reviews_per_movie=5)
        
        if not movie_data:
            print("✗ 未能爬取到电影数据")
            return
        
        # 保存数据
        file_paths = crawler.save_data(movie_data, output_dir)
        
        # 显示结果
        print("\n" + "=" * 80)
        print("✅ 影史评分前250电影语料库构建完成！")
        print("=" * 80)
        print(f"\n生成的文件:")
        print(f"  1. 完整情感语料库 (JSON): {file_paths['json_corpus']}")
        print(f"  2. 增强版电影数据 (CSV): {file_paths['enhanced_csv']} ← 推荐使用")
        print(f"  3. 排名信息 (CSV): {file_paths['ranking']}")
        print(f"  4. 基础数据 (CSV): {file_paths['csv_data']}")
        print(f"  5. 影评数据 (JSON): {file_paths['reviews']}")
        print(f"  6. 情感向量 (CSV): {file_paths['emotion_vectors']}")
        
        # 显示统计信息
        print(f"\n📊 语料库统计:")
        print("-" * 40)
        print(f"电影总数: {len(movie_data)}")
        
        total_reviews = sum(len(m.get('reviews', [])) for m in movie_data)
        print(f"影评总数: {total_reviews}")
        print(f"平均影评数/电影: {total_reviews/len(movie_data):.1f}")
        
        # 评分统计
        avg_rating = np.mean([m['vote_average'] for m in movie_data])
        avg_votes = np.mean([m['vote_count'] for m in movie_data])
        print(f"平均评分: {avg_rating:.2f}/10")
        print(f"平均投票数: {avg_votes:,.0f}")
        
        # 情感统计
        emotion_counter = {}
        for movie in movie_data:
            for emotion in movie.get('dominant_emotions', []):
                emotion_counter[emotion] = emotion_counter.get(emotion, 0) + 1
        
        print(f"\n主导情感分布 (前5):")
        for emotion, count in sorted(emotion_counter.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(movie_data)) * 100
            print(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # 显示前10名电影
        print(f"\n🏆 影史评分前10名:")
        sorted_movies = sorted(movie_data, key=lambda x: x.get('tmdb_top_rated_rank', 0))
        for i, movie in enumerate(sorted_movies[:10], 1):
            print(f"  {i}. 《{movie['title']}》 (评分: {movie['vote_average']}/10)")
        
        # 下一步提示
        print("\n" + "=" * 80)
        print("🚀 下一步:")
        print("=" * 80)
        print("1. 使用增强版电影数据运行电影推荐程序:")
        print(f"   python emotion_movie_recommender.py")
        print()
        print("2. 在推荐程序中，使用以下文件路径:")
        print(f"   enhanced_csv = \"{file_paths['enhanced_csv']}\"")
        print(f"   或")
        print(f"   json_corpus = \"{file_paths['json_corpus']}\"")
        print()
        print("3. 如需重新构建语料库，再次运行此程序")
        
    except Exception as e:
        print(f"\n✗ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查依赖
    try:
        import requests
        import pandas
        print("✓ 依赖检查通过")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install requests pandas numpy")
        exit(1)
    
    main()
