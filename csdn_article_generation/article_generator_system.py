import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import requests
import time
from pathlib import Path
import warnings
import os
import pickle
import re
warnings.filterwarnings('ignore')

# 使用轻量级的Sentence Transformer模型
class LocalEmbedder:
    def __init__(self, model_name: str = "./qwen3-embed-0.6b"):
        try:
            from sentence_transformers import SentenceTransformer
            # 使用轻量级模型，减少内存占用
            self.model = SentenceTransformer(model_name)
            print(f"本地嵌入模型 {model_name} 加载成功")
        except Exception as e:
            print(f"加载本地模型失败: {e}")
            self.model = None
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            # 备用方案：随机向量
            return np.random.randn(len(texts), 384)
        
        # Sentence Transformers 自动处理批处理和标准化
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

# 向量存储和检索 - 使用npy格式优化存储
class VectorStore:
    def __init__(self, embedder, index_dir: str = "vector_index"):
        self.embedder = embedder
        self.embeddings = None
        self.questions = []
        self.answers = []
        self.data = []
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
    
    def load_json_data(self, json_path: str):
        """加载新的JSON格式的QA数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if 'paraname' in item and 'content' in item:
                self.questions.append(item['paraname'])
                self.answers.append(item['content'])
                self.data.append(item)
        
        print(f"加载了 {len(self.questions)} 个段落对")
    
    def save_index(self):
        """保存向量索引到npy文件"""
        if self.embeddings is None:
            print("没有索引可保存")
            return
        
        # 保存嵌入向量
        np.save(os.path.join(self.index_dir, 'embeddings.npy'), self.embeddings)
        
        # 保存文本数据
        index_data = {
            'questions': self.questions,
            'answers': self.answers,
            'data': self.data
        }
        
        with open(os.path.join(self.index_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"向量索引已保存到 {self.index_dir}")
    
    def load_index(self) -> bool:
        """从npy文件加载向量索引"""
        embeddings_path = os.path.join(self.index_dir, 'embeddings.npy')
        metadata_path = os.path.join(self.index_dir, 'metadata.pkl')
        
        if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
            print("索引文件不存在")
            return False
        
        try:
            # 加载嵌入向量
            self.embeddings = np.load(embeddings_path)
            
            # 加载元数据
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.questions = metadata['questions']
            self.answers = metadata['answers']
            self.data = metadata['data']
            
            print(f"从 {self.index_dir} 加载了向量索引，包含 {len(self.questions)} 个段落对")
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def build_index(self, force_rebuild: bool = False, batch_size: int = 32):
        """构建向量索引，支持批量处理减少内存压力"""
        if not force_rebuild and self.load_index():
            return
        
        if not self.questions:
            print("没有数据可索引")
            return
        
        print("开始构建向量索引...")
        
        # 分批处理以减少内存使用
        all_embeddings = []
        for i in range(0, len(self.questions), batch_size):
            batch_texts = self.questions[i:i + batch_size]
            batch_embeddings = self.embedder.embed(batch_texts)
            all_embeddings.append(batch_embeddings)
            print(f"已处理 {min(i + batch_size, len(self.questions))}/{len(self.questions)} 条数据")
        
        self.embeddings = np.vstack(all_embeddings)
        print(f"向量索引构建完成，维度: {self.embeddings.shape}")
        
        # 保存新构建的索引
        self.save_index()
    
    def search_with_threshold(self, query: str, similarity_threshold: float = 0.3, max_results: int = 100) -> List[Dict]:
        """基于相似度阈值搜索，而不是固定top_k"""
        if self.embeddings is None:
            self.build_index()
        
        query_embedding = self.embedder.embed([query])[0]
        
        # 使用余弦相似度
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norm = np.linalg.norm(self.embeddings, axis=1)
        
        # 避免除零错误
        similarities = np.dot(self.embeddings, query_embedding) / (
            embeddings_norm * query_norm + 1e-8
        )
        
        # 获取所有超过阈值的索引
        above_threshold_indices = np.where(similarities >= similarity_threshold)[0]
        
        # 按相似度排序并限制最大数量
        sorted_indices = above_threshold_indices[np.argsort(similarities[above_threshold_indices])[::-1]]
        selected_indices = sorted_indices[:max_results]
        
        results = []
        for idx in selected_indices:
            results.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'similarity': similarities[idx],
                'original_data': self.data[idx]
            })
        
        print(f"找到 {len(results)} 个相似度 >= {similarity_threshold} 的结果")
        return results

# DeepSeek API客户端（添加重试和超时处理）
class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.strip()  # 修复base_url末尾空格问题
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print(f"DeepSeek客户端初始化完成，API密钥: {api_key[:8]}...")  # 调试信息
    
    def chat_completion(self, messages: List[Dict], model: str = "deepseek-chat", 
                       temperature: float = 0.7, max_tokens: int = 2000, max_retries: int = 3) -> str:
        """调用DeepSeek API，添加重试机制"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # 调试信息：显示请求摘要
        user_message = next((msg['content'][:50] + '...' for msg in messages if msg['role'] == 'user'), '')
        print(f"正在调用DeepSeek API，用户消息: {user_message}")
        
        for attempt in range(max_retries):
            try:
                print(f"API请求尝试 {attempt + 1}/{max_retries}...")
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=90  # 增加超时时间
                )
                
                response_time = time.time() - start_time
                print(f"API响应时间: {response_time:.2f}秒，状态码: {response.status_code}")
                
                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"]
                print("API调用成功！")
                return result
                
            except requests.exceptions.Timeout:
                print(f"⚠️ API请求超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"等待 {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    return "错误: API请求超时，请检查网络连接"
                    
            except requests.exceptions.ConnectionError:
                print(f"⚠️ 网络连接错误 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return "错误: 网络连接失败，请检查网络设置"
                    
            except Exception as e:
                print(f"❌ API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return f"错误: {str(e)}"
        
        return "错误: API调用失败，超过最大重试次数"

# 多文件向量存储管理器
class MultiVectorStoreManager:
    def __init__(self, json_files: List[str], embedder, base_index_dir: str = "vector_indexes"):
        self.json_files = json_files
        self.embedder = embedder
        self.base_index_dir = base_index_dir
        self.stores = {}
        os.makedirs(base_index_dir, exist_ok=True)
        
        # 为每个JSON文件创建对应的向量存储
        for json_file in json_files:
            name = os.path.splitext(os.path.basename(json_file))[0]
            index_dir = os.path.join(base_index_dir, name)
            self.stores[name] = VectorStore(embedder, index_dir)
    
    def load_and_build_all_indexes(self, force_rebuild: bool = False):
        """加载并构建所有向量索引"""
        for name, store in self.stores.items():
            json_file = f"{name}.json"
            if os.path.exists(json_file):
                print(f"正在处理 {json_file}...")
                store.load_json_data(json_file)
                store.build_index(force_rebuild=force_rebuild, batch_size=16)
            else:
                print(f"警告: 找不到文件 {json_file}")
    
    def search_all_stores(self, query: str, similarity_threshold: float = 0.3, max_results_per_store: int = 100) -> List[Dict]:
        """在所有存储中搜索"""
        all_results = []
        for name, store in self.stores.items():
            print(f"在 {name} 中搜索...")
            results = store.search_with_threshold(query, similarity_threshold, max_results_per_store)
            # 为每个结果添加来源信息
            for result in results:
                result['source'] = name
            all_results.extend(results)
        return all_results

# 文章生成系统（优化内存使用）
class ArticleGenerator:
    def __init__(self, json_files: List[str], deepseek_api_key: str, 
                 similarity_threshold: float = 0.3,  # 相似度阈值
                 max_iterations: int = 100,  # 最大迭代次数增加到100
                 max_context_length: int = 100000,  # 上下文长度上限
                 base_index_dir: str = "vector_indexes"):
        # 使用更轻量的模型
        self.embedder = LocalEmbedder("./qwen3-embed-0.6b")
        self.vector_manager = MultiVectorStoreManager(json_files, self.embedder, base_index_dir)
        self.deepseek_client = DeepSeekClient(deepseek_api_key)

        self.info_insufficient_flag = True # 新增：用于记录是否曾判断信息不足
        
        # 加载数据
        print("正在加载数据...")
        self.vector_manager.load_and_build_all_indexes()
        
        # 动态分析文档类型
        self.document_info = self._analyze_document_types()
        
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations
        self.max_context_length = max_context_length
        self.collected_context = []  # 存储完整上下文（包含content）
        self.relevant_paranames = set()  # 只存储相关的paraname，用于去重
        self.intermediate_response = ""  # 新增：中间回答
        self.all_searched_paranames = []  # 新增：存储所有搜索到的paraname，无论是否相关
        self.insufficient_count = 0  # 新增：记录连续信息不足的次数
        self.user_input = ""  # 新增：存储用户的原始输入
        print("文章生成器初始化完成")
    
    def _analyze_document_types(self) -> Dict[str, str]:
        """动态分析文档类型，基于文档名和文档内容"""
        doc_info = {}
        
        for name, store in self.vector_manager.stores.items():
            if not store.data:
                doc_info[name] = "未知类型的技术文档"
                continue
            
            # 从文档内容中提取样本进行分析
            sample_data = store.data[:min(10, len(store.data))]  # 取前10个样本
            
            # 提取paraname和content中的关键词
            sample_paranames = [item.get('paraname', '') for item in sample_data]
            sample_contents = [item.get('content', '') for item in sample_data]
            
            # 构造分析提示
            sample_text = "\n".join([
                f"段落标题: {pn[:100]}...\n内容预览: {cont[:200]}..."
                for pn, cont in zip(sample_paranames, sample_contents)
            ])
            
            prompt = f"""基于以下技术文档的样本内容和文件名，分析这是什么类型的技术文档：

文件名: {name}

样本内容:
{sample_text}

请分析并回答：
1. 这是什么领域的技术文档？（如：无线通信、网络协议、软件开发、人工智能、数据库等）
2. 涉及什么具体技术标准或规范？（如：WiFi 6、5G、TCP/IP、HTTP/2、SQL等）
3. 文档的主要内容是什么？（如：MAC层协议、物理层规范、安全机制、API设计、算法实现等）
4. 该文档的用途是什么？（如：技术规范、用户手册、开发指南、参考文档等）

请以JSON格式返回：
{{
    "domain": "技术领域",
    "standard": "涉及的技术标准",
    "content_type": "文档主要内容类型",
    "purpose": "文档用途",
    "description": "简要描述"
}}

只返回JSON格式，不要其他内容："""
            
            messages = [
                {"role": "system", "content": "你是一个技术文档分析专家，能够准确识别技术文档的类型和领域。"},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = self.deepseek_client.chat_completion(messages, temperature=0.3, max_tokens=400)
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    doc_info[name] = f"{analysis.get('domain', '技术文档')} - {analysis.get('standard', '未知标准')} - {analysis.get('content_type', '未知内容类型')} - {analysis.get('purpose', '参考文档')}"
                    print(f"文档 {name} 类型分析: {doc_info[name]}")
                else:
                    doc_info[name] = f"技术文档 - {name}"
            except Exception as e:
                print(f"分析文档 {name} 类型失败: {e}")
                doc_info[name] = f"技术文档 - {name}"
        
        return doc_info
    
    def _get_document_context(self) -> str:
        """获取文档类型上下文信息"""
        context = "检索的文档类型包括：\n"
        for name, doc_desc in self.document_info.items():
            context += f"- {name}: {doc_desc}\n"
        return context

    def clarify_user_intent(self, user_input: str) -> str:
        """澄清用户意图，直到AI确认明确"""
        print("开始澄清用户意图...")
        
        # 初始澄清循环
        clarification_round = 0
        current_input = user_input
        
        while True:
            clarification_round += 1
            
            # 执行初步搜索
            search_results = self.vector_manager.search_all_stores(
                current_input, 
                similarity_threshold=self.similarity_threshold,
                max_results_per_store=50
            )
            
            # 提取搜索结果的标题
            search_titles = [result['question'] for result in search_results[:10]]  # 取前10个
            titles_text = "\n".join([f"- {title[:100]}..." for title in search_titles]) if search_titles else "无相关结果"
            
            # 构造澄清提示
            prompt = f"""你是一个专业的技术文档分析助手。请分析用户的需求并判断是否明确。

{self._get_document_context()}

当前用户输入: {current_input}

基于初步搜索，相关标题包括:
{titles_text}

请判断以下几点：
1. 用户的意图是否明确？（是/否）
2. 如果不明确，请提出1-2个具体问题来帮助澄清用户需求
3. 如果明确，请简要说明你理解的用户需求

请以JSON格式回复:
{{
    "intent_clear": true/false,
    "clarification_questions": ["问题1", "问题2"],
    "understood_intent": "你理解的用户需求"
}}

只返回JSON格式的响应："""
            
            messages = [
                {"role": "system", "content": "你是一个专业的技术文档分析助手，能够准确判断用户意图是否明确。"},
                {"role": "user", "content": prompt}
            ]
            
            print(f"正在进行第 {clarification_round} 轮意图澄清...")
            response = self.deepseek_client.chat_completion(messages, temperature=0.3, max_tokens=500)
            
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    intent_clear = analysis.get('intent_clear', False)
                    clarification_questions = analysis.get('clarification_questions', [])
                    understood_intent = analysis.get('understood_intent', '')
                    
                    if intent_clear:
                        print(f"✅ 用户意图已明确: {understood_intent}")
                        return understood_intent
                    elif clarification_questions:
                        print("需要进一步澄清用户意图:")
                        for i, question in enumerate(clarification_questions, 1):
                            print(f"  {i}. {question}")
                        
                        # 询问用户
                        print("\n为了更好地理解您的需求，请回答以上问题:")
                        user_response = input("您的回答: ").strip()
                        
                        if user_response:
                            # 更新当前输入，用于下一轮搜索和分析
                            current_input = f"{user_input}\n\n用户补充说明: {user_response}"
                        else:
                            print("未收到有效回答，继续使用原始输入...")
                    else:
                        print("❌ 无法解析意图分析结果，继续使用原始输入...")
                        return user_input
                else:
                    print("❌ 无法解析意图分析结果，继续使用原始输入...")
                    return user_input
            except Exception as e:
                print(f"❌ 解析意图分析结果失败: {e}")
                return user_input
            
            # 限制澄清轮数，避免无限循环
            if clarification_round >= 3:
                print("已达到最大澄清轮数，继续使用当前理解的意图...")
                return current_input

    def generate_search_query(self, user_input: str, context: List[Dict], intermediate_response: str) -> str:
        """生成搜索查询，引导AI关注当前缺少的信息类型"""
        
        # 获取已有的 paranames（相关的结果）
        existing_paranames = [ctx['original_data'].get('paraname', '') for ctx in context if 'original_data' in ctx]
        existing_paranames_preview = "\n".join([f"- {p[:100]}..." for p in existing_paranames[-5:]]) if existing_paranames else "无"
        
        # 获取所有搜索到的 paranames（包括不相关的）
        all_searched_preview = "\n".join([f"- {p[:100]}..." for p in self.all_searched_paranames[-10:]]) if self.all_searched_paranames else "无"
    
        # 提示AI分析当前信息缺口并生成针对性查询
        prompt = f"""你是专业的技术信息检索助手。请根据用户需求、已有信息和中间回答，生成一个**精准的补全式搜索查询**。
        
{self._get_document_context()}

用户原始需求: {user_input}
        
已有相关信息标题:
{existing_paranames_preview}
        
当前中间回答:
{intermediate_response[:500] + '...' if len(intermediate_response) > 500 else intermediate_response}
        
历史上所有搜索到的标题（可能相关也可能不相关）:
{all_searched_preview}
        
必须思考：
1. 当前已有信息主要集中在哪些方面？
2. 已有信息标题意味着什么全景？用户的需求位于全景之下的哪个地方？
3. 全景内还可能有什么？寻找视野之外的全景！
4. 中间回答是否已经足够全面？还有哪些缺失的部分需要补充？
5. 历史上所有搜索到的标题中，哪些可能是被误判为不相关的但实际有用的？
        
💡 必须这样做：
- 你实际生成假设的段落标题或者对假设的段落内内容的总结性提问，便于向量检索匹配。
- 你可以生成整个文章的任意章节的段落标题或者对任何段落内内容的总结性提问，所以关键词都很可能和用户的问题或者已有的标题不一样！只要它们对用户有利！
- 你实际在角色扮演，写文章起段落标题，或者资深专家对段落内内容的总结性提问，已有信息是真实的例子但是很可能是表面的例子，你需要想象力！
        
输出要求：
- 只返回一个新的搜索查询语句，不需要解释。
- 查询要具体、专业，聚焦于某个技术点或方向。
"""
    
        messages = [
            {"role": "system", "content": "你是一个专业的技术信息检索助手，擅长分析信息缺口并生成精准的技术搜索查询。"},
            {"role": "user", "content": prompt}
        ]
    
        print("正在生成补全式搜索查询...")
        query = self.deepseek_client.chat_completion(messages, temperature=0.5, max_tokens=120)
        return query.strip('"\'')    

    def filter_relevant_content(self, search_results: List[Dict], user_input: str, intermediate_response: str) -> List[Dict]:
        if not search_results:
            return []
    
        results_text = "\n\n".join([
            f"结果 {i+1} (相似度: {result['similarity']:.3f}, 来源: {result['source']}):\n"
            f"段落标题: {result['question']}"
            for i, result in enumerate(search_results)
        ])
    
        prompt = f"""请分析以下搜索结果的段落标题是否与用户需求相关，并筛选出真正相关的段落。

{self._get_document_context()}

用户需求: {user_input}
    
当前中间回答:
{intermediate_response[:500] + '...' if len(intermediate_response) > 500 else intermediate_response}
    
搜索结果段落标题:
{results_text}
    
⚠️ 注意事项：
- 相似度分数仅供参考，请勿仅依据相似度决定。
- 必须确保段落主题、用户需求、数据原始来源三者一致才算作相关。
- 如果某个段落虽然关键词匹配但来源不一致或语义不符，请排除。
- 考虑当前中间回答的内容，判断是否真的需要这些新信息。
- 检索的对象是技术规范文档中的段落，需要基于技术事实判断相关性。
- 根据文档类型信息判断内容的相关性（参考上面的文档类型信息）

例外情况：
- 如果看过去都不相关，这个时候你可以先不顾用户需求，把你感兴趣的段落认为相关，下一轮会给你看段落内的具体内容。
    
请分析每个搜索结果的相关性，并返回一个JSON格式的响应，包含:
1. relevant_indices: 相关结果的索引列表（从1开始）
2. reasoning: 简要说明筛选理由
    
只返回JSON格式的响应，不要有其他内容："""
    
        messages = [
            {"role": "system", "content": "你是一个专业的技术内容筛选专家，能够准确判断信息相关性。"},
            {"role": "user", "content": prompt}
        ]
    
        print("🔍 正在使用AI判断内容相关性...")
        response = self.deepseek_client.chat_completion(messages, temperature=0.2, max_tokens=800)
    
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                filter_result = json.loads(json_match.group())
                relevant_indices = filter_result.get('relevant_indices', [])
                relevant_results = [search_results[idx - 1] for idx in relevant_indices if 1 <= idx <= len(search_results)]
                print(f"✅ AI筛选结果: 从 {len(search_results)} 个结果中筛选出 {len(relevant_results)} 个相关结果")
                return relevant_results
        except Exception as e:
            print(f"❌ 解析AI筛选结果失败: {e}")
    
        return search_results
    
    def should_stop_search(self, iteration: int, total_context_length: int) -> bool:
        """判断是否应该停止搜索"""
        if iteration >= self.max_iterations:
            print(f"达到最大迭代次数 ({self.max_iterations})，停止搜索")
            return True
        
        if total_context_length >= self.max_context_length:
            print(f"上下文长度达到上限 ({total_context_length}字符)，停止搜索")
            return True
        
        # 可以添加其他停止条件，如连续几轮没有新内容等
        return False
    
    def generate_article(self, user_input: str) -> Tuple[str, str, List[str]]:
        """生成完整的文章，返回文章内容、标题和使用的paraname列表"""
        print(f"\n开始处理用户输入: '{user_input}'")
        
        # 首先澄清用户意图
        clarified_intent = self.clarify_user_intent(user_input)
        self.user_input = clarified_intent  # 保存澄清后的意图

        # 重置上下文
        self.collected_context = []
        self.relevant_paranames = set()
        self.intermediate_response = ""
        self.all_searched_paranames = []  # 重置所有搜索到的paraname列表
        self.insufficient_count = 0  # 重置信息不足计数器
        iteration = 0
        total_context_length = 0
        
        while not self.should_stop_search(iteration, total_context_length):
            iteration += 1
            print(f"\n开始第 {iteration} 轮搜索...")
            
            if iteration == 1:
                # 第一轮使用澄清后的意图
                query = clarified_intent
            else:
                # 后续轮次生成新的搜索查询
                query = self.generate_search_query(clarified_intent, self.collected_context, self.intermediate_response)
                print(f"生成的搜索查询: '{query}'")
                
                if not query or len(query) < 3:
                    print("搜索查询无效，停止搜索")
                    break
            
            # 执行基于阈值的搜索
            search_results = self.vector_manager.search_all_stores(
                query, 
                similarity_threshold=self.similarity_threshold,
                max_results_per_store=100
            )
            
            if not search_results:
                print("没有找到相关结果，停止搜索")
                break
            
            # 将所有搜索到的paraname添加到历史记录中
            for result in search_results:
                paraname = result['original_data'].get('paraname', '')
                if paraname and paraname not in self.all_searched_paranames:
                    self.all_searched_paranames.append(paraname)
            
            # 使用AI筛选相关的内容（只基于paraname判断）
            relevant_results = self.filter_relevant_content(search_results, clarified_intent, self.intermediate_response)
            
            if not relevant_results:
                print("AI筛选后没有相关结果，停止搜索")
                break
            
            # 去重并添加到收集的上下文（基于paraname去重）
            new_results = []
            
            for result in relevant_results:
                paraname = result['original_data'].get('paraname', '')
                if paraname and paraname not in self.relevant_paranames:
                    self.relevant_paranames.add(paraname)
                    new_results.append(result)
                    total_context_length += len(result['question']) + len(result['answer'])
            
            if not new_results:
                print("没有新的相关内容，停止搜索")
                break
            
            self.collected_context.extend(new_results)
            print(f"新增 {len(new_results)} 个相关结果，总上下文长度: {total_context_length} 字符")
            
            # 生成中间回答
            self.intermediate_response = self._generate_intermediate_response(clarified_intent, new_results)
            print(f"生成中间回答: {self.intermediate_response[:100]}...")
            
            # 检查是否信息足够（由AI判断）
            if self.is_information_sufficient(clarified_intent):
                print("AI判断信息已足够，停止搜索")
                break
        
        # 生成最终文章
        article, title = self._generate_final_article(clarified_intent)
        return article, title, list(self.relevant_paranames)
    
    def _generate_intermediate_response(self, user_input: str, new_results: List[Dict]) -> str:
        """生成中间回答，带上之前的中间回答作为上下文"""
        if not new_results:
            return self.intermediate_response  # 没有新内容则保持上一轮的回答
    
        # 准备新获取的信息
        context_str = "\n\n".join([
            f"### 相关信息 {i+1} (相似度: {ctx['similarity']:.3f}, 来源: {ctx['source']})\n"
            f"**段落标题**: {ctx['original_data']['paraname']}\n"
            f"**内容**: {ctx['answer'][:500]}..."
            for i, ctx in enumerate(new_results)
        ])
    
        # 构造 prompt，带上上一轮的中间回答
        prompt = f"""基于以下新获取的信息和用户原始需求，结合你之前的中间回答，生成一个新的中间回答以替换你之前的中间回答：

{self._get_document_context()}
    
**用户原始需求**: {user_input}
    
**你之前的中间回答**:
{self.intermediate_response[:1000] + '...' if len(self.intermediate_response) > 1000 else self.intermediate_response}
    
**新获取的信息**:
{context_str}
    
**要求**:
1. 回答要延续之前的逻辑，不要重复也不要矛盾
2. 整合新信息，使内容更完整
3. 保持专业性和准确性
4. 字数控制在300-500字之间
5. 基于技术事实进行陈述，不要添加推测内容
6. 结合文档类型特点进行专业分析
    
请直接输出中间回答："""
    
        messages = [
            {"role": "system", "content": "你是一位资深技术专家，擅长整合信息并给出专业的中间回答。"},
            {"role": "user", "content": prompt}
        ]
    
        print("正在生成中间回答...")
        response = self.deepseek_client.chat_completion(messages, temperature=0.7, max_tokens=1000)
        return response

    def is_information_sufficient(self, user_input: str) -> bool:
        """由AI判断收集的信息是否足够生成文章，只基于paraname判断"""
        if not self.collected_context:
            return False
        
        # 准备上下文预览，只使用paraname
        context_preview = "\n".join([
            f"- {ctx['question'][:50]}... (相似度: {ctx['similarity']:.3f}, 来源: {ctx['source']})"
            for ctx in self.collected_context[-5:]  # 显示最后5个结果
        ])
        
        prompt = f"""基于当前收集的段落标题和中间回答，判断是否足够生成一篇关于"{user_input}"的完整文章。

{self._get_document_context()}

已收集的段落标题概要:
{context_preview}

当前中间回答:
{self.intermediate_response[:500] + '...' if len(self.intermediate_response) > 500 else self.intermediate_response}

总共收集了 {len(self.collected_context)} 条相关信息。

请判断是否还需要继续搜索更多信息，还是已经足够生成高质量文章。返回JSON格式响应:
{{
    "sufficient": true/false,
    "reason": "简要说明理由"
}}

只返回JSON格式的响应："""
        
        messages = [
            {"role": "system", "content": "你是一个专业的技术内容评估专家，能够准确判断信息是否足够生成高质量文章。"},
            {"role": "user", "content": prompt}
        ]
        
        print("正在评估信息是否足够...")
        response = self.deepseek_client.chat_completion(messages, temperature=0.2, max_tokens=300)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                sufficient = assessment.get('sufficient', False)
                reason = assessment.get('reason', '')
                
                print(f"信息充足性评估: {'足够' if sufficient else '不足'} - {reason}")

                if not sufficient:
                    self.info_insufficient_flag = True  # 设置标志位
                    self.insufficient_count += 1  # 增加计数器
                    
                    # 如果连续三次信息不足，询问用户是否继续
                    if self.insufficient_count >= 3:
                        print("\n⚠️ 连续三次判断信息不足，需要用户确认是否立即生成文章")
                        user_choice = input("是否立即生成文章？(y/n): ").strip().lower()
                        if user_choice in ['y', 'yes', '是']:
                            print("用户选择立即生成文章")
                            self.insufficient_count = 0  # 重置计数器
                            return True  # 认为信息足够
                        else:
                            print("用户选择不立即生成文章")
                            return False  # 确实信息不足
                else:
                    self.info_insufficient_flag = False  # 设置标志位
                    self.insufficient_count = 0  # 重置计数器

                return sufficient
        except Exception as e:
            print(f"解析信息充足性评估失败: {e}")
        
        # 默认继续搜索
        return False
    
    def _extract_title_from_article(self, article: str) -> str:
        """从文章内容中提取标题"""
        # 查找Markdown格式的标题
        title_match = re.search(r'^#\s+(.+)$', article, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        
        # 如果没有找到Markdown标题，尝试提取第一行作为标题
        first_line = article.split('\n')[0].strip()
        if first_line and len(first_line) < 100:  # 确保不是过长的文本
            return first_line
        
        # 如果都无法提取，返回默认标题
        return "生成的文章"
    
    def _generate_final_article(self, user_input: str) -> Tuple[str, str]:
        """生成最终的文章内容，并返回内容和标题"""
        print(f"\n正在生成最终文章，使用 {len(self.collected_context)} 个上下文片段...")

        if not self.collected_context:
            # 没有可用的上下文，提前终止
            message = (
                "# 无法生成文章\n\n"
                "抱歉，根据当前知识库未能检索到与您的输入“{}”相关的任何内容。\n\n"
                "请尝试提供更具体或不同的关键词。"
            ).format(user_input)
            return message, f"未找到相关内容-{user_input}"       

        # 按相似度排序并选择最佳上下文
        sorted_context = sorted(self.collected_context, key=lambda x: x['similarity'], reverse=True)
        
        # 限制上下文长度但确保质量
        max_context_used = min(10, len(sorted_context))  # 最多使用10个最佳上下文
        best_context = sorted_context[:max_context_used]
        
        context_str = "\n\n".join([
            f"### 相关信息 {i+1} (相似度: {ctx['similarity']:.3f}, 来源: {ctx['source']})\n"
            f"**段落标题**: {ctx['original_data']['paraname']}\n"
            f"**内容**: {ctx['answer'][:500]}..."
            for i, ctx in enumerate(best_context)
        ])
        
        prompt = f"""基于以下参考信息和中间回答，撰写一篇专业、结构完整的技术文章：

{self._get_document_context()}

**文章主题**: {user_input}

**中间回答**:
{self.intermediate_response}

**参考信息**:
{context_str}

**写作要求**:
1. 文章标题要有吸引力, 又不失专业性
2. 结构包含：遵循金字塔思维模型，以故事或者疑问入手，并可以先给出答案，再核心内容（分多个小节）、最后呼应开头
3. 内容深度和技术准确性并重
4. 字数1500-2500字
5. 使用Markdown格式，包含适当的标题层级
6. 确保逻辑连贯，信息准确
7. 可以适当扩展和补充相关知识, 只能补充技术事实性知识，其他不要补充！
8. 不要直接复制中间回答，而是将其作为基础进行润色
9. 基于技术事实进行陈述，结合你已有的知识和检索到的信息
10. 检索到的信息是技术规范文档中的段落，需要确保内容的准确性和专业性
11. 结合文档类型特点进行专业分析和阐述
12. 在合适和可行的地方可以考虑用mermaid可视化，宁缺毋滥，不强求


请直接输出完整的文章内容："""

        messages = [
            {"role": "system", "content": "你是一位资深技术作家，擅长撰写深度技术文章，能够将复杂的技术概念讲解得清晰易懂。"},
            {"role": "user", "content": prompt}
        ]
        
        print("正在调用DeepSeek API生成高质量文章...")
        article = self.deepseek_client.chat_completion(messages, temperature=0.8, max_tokens=4000)
        
        # 从文章中提取标题
        title = self._extract_title_from_article(article)
        print(f"提取的文章标题: {title}")
        
        return article, title

# 主程序
def main():
    # 配置参数
    JSON_FILES =  ["qa_data_mac2024.new.suffix.json", "11ax.suffix.json", "qa_data_11be.suffix.json"]#, "qa_data.mac-qa.json"] #["qa_data_mac2024.new.json"]
    #JSON_FILES =  ["qa_data.mac-qa.json"] #["qa_data_mac2024.new.json"]
    #JSON_FILES =  ["qa_data_mac2024.new.suffix.json"]#, "qa_data.mac-qa.json"] #["qa_data_mac2024.new.json"]
    #JSON_FILES = ["qa_data_mac2024.json"]
    #JSON_FILES = ["qa_data_802.11bn-pdt-mac-dbe-part-2.json","qa_data_802.11bn-pdt-mac-dbe.json"]
    #JSON_FILES = ["qa_data_802.11bn_pdt-mac-co-tdma-part-1.json","qa_data_802.11bn_pdt-mac-co-tdma-part-2.json","qa_data_802.11bn_pdt-mac-co-tdma-part-3.json"]
    #JSON_FILES = ["qa_data_802.11bn_pdt-mac-on-seamless-roaming-part-1.json", "qa_data_802.11bn_pdt-mac-on-seamless-roaming-part-2.json", "qa_data_802.11bn_pdt-mac-on-seamless-roaming-part-3.json", "qa_data_802.11bn_pdt-mac-on-seamless-roaming-part-4.json", "qa_data_802.11bn_pdt-mac-on-seamless-roaming-part-5.json"]  # 替换为你的JSON文件列表
    #JSON_FILES =  ["CUDA_C_Programming_Guide_v12.1-19-498.json"] #["qa_data_mac2024.new.json"]
    #JSON_FILES =  ["BT_Core_specification_v5.3-181-3085.json"] #["qa_data_mac2024.new.json"]
    #JSON_FILES =  ["USB_3.0_R1.0-29-440.json", "USB4_1.0_with_errata_through_20201015-CLEAN-36-560.json"] #["qa_data_mac2024.new.json"]
    #JSON_FILES =  ["USB_3.0_R1.0-29-440.json", "USB4_1.0_with_errata_through_20201015-CLEAN-36-560.json", "PCI_Express_Base_Specification_Revision_6.0-99-1520.json"] #["qa_data_mac2024.new.json"]
    #JSON_FILES =  ["PCI_Express_Base_Specification_Revision_6.0-99-1520.json"] #["qa_data_mac2024.new.json"]
    #JSON_FILES =  ["PCI_Express_Base_Specification_Revision_6.0-99-1520.json", "PCI_Express_Base_Specification_Revision_5.0_Version_1.0-89-1210.json"] #["qa_data_mac2024.new.json"]
    DEEPSEEK_API_KEY = "sk-e86cebed80e445d8a3b1a6e715d6d1f2"  # 替换为你的API密钥
    BASE_INDEX_DIR = "vector_indexes"  # 向量索引保存目录
    
    # 检查依赖
    try:
        import sentence_transformers
    except ImportError:
        print("请安装sentence-transformers: pip install sentence-transformers")
        return
    
    print("=" * 60)
    print("高级文章生成系统启动")
    print("=" * 60)
    
    # 初始化文章生成器
    generator = ArticleGenerator(
        JSON_FILES, 
        DEEPSEEK_API_KEY, 
        similarity_threshold=0.3,  # 相似度阈值
        max_iterations=100,        # 最大迭代次数
        max_context_length=100000, # 上下文长度上限
        base_index_dir=BASE_INDEX_DIR
    )
    
    # 用户输入
    user_input = input("请输入您想要生成文章的主题或问题: ")
    
    if not user_input.strip():
        print("输入不能为空！")
        return
    
    # 生成文章
    start_time = time.time()
    article, title, used_paranames = generator.generate_article(user_input)
    end_time = time.time()
    
    print(f"\n文章生成完成，总耗时: {end_time - start_time:.2f}秒")
    
    # 显示使用的paraname
    print("\n=== 使用的段落标题 (paraname) ===")
    for i, paraname in enumerate(used_paranames, 1):
        print(f"{i}. {paraname}")
    print(f"\n总共使用了 {len(used_paranames)} 个段落标题")
    
    # 使用文章标题作为文件名
    # 清理标题中的非法文件名字符
    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
    safe_title = safe_title.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_title = safe_title[:100]  # 限制文件名长度
    
    output_path = f"{safe_title}.md"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(article)
    
    print(f"\n文章已生成并保存到: {output_path}")
    print("\n=== 文章预览 ===\n")
    print(article[:800] + "..." if len(article) > 800 else article)
    print(f"\n文章总长度: {len(article)} 字符")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断了操作，程序已退出。")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
