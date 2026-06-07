# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import time
import random
import traceback
from openai import OpenAI

class GenerationPausedException(Exception):
    """Custom exception raised when generation is paused mid-chapter."""
    pass

class NovelGenerator:
    def __init__(self, project_path, api_key, api_base, model_name, config=None):
        self.project_path = project_path
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.config = config or {}
        
        # Paths
        self.startup_file = os.path.join(self.project_path, "docs", "startup.md")
        self.queue_file = os.path.join(self.project_path, "tasks_queue.json")
        self.memory_file = os.path.join(self.project_path, "world_memory.json")
        self.novel_file = os.path.join(self.project_path, "master_novel.md")
        self.log_file = os.path.join(self.project_path, "generation.log")
        
        # Configure variables from config
        self.total_chapters = int(self.config.get("total_chapters", 100))
        self.min_words = int(self.config.get("min_words", 3000))
        self.max_words = int(self.config.get("max_words", 5000))
        self.style = self.config.get("style", "科幻/硬科幻")
        self.title = self.config.get("title", "未命名小说")
        self.temperature = float(self.config.get("temperature", 0.7))
        self.ref_text = self.config.get("ref_text", "")
        
        # Create client
        if self.api_base:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
    def log(self, message):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{timestamp} {message}"
        print(line)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"Error writing log: {e}")
            
    def check_abort(self):
        """Check if pause is requested and raise GenerationPausedException."""
        if hasattr(self, "check_stop_callback") and self.check_stop_callback and self.check_stop_callback():
            raise GenerationPausedException("Generation paused cooperatively.")

    def call_llm(self, prompt, temperature=0.7, max_tokens=None, max_retries=6):
        """Call LLM with exponential backoff on failure."""
        self.check_abort()
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(max_retries):
            self.check_abort()
            try:
                # Add optional max_tokens if supported, otherwise let it default
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": 60.0
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                    
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM returned empty or None content")
                return content
            except Exception as e:
                wait_time = (2 ** attempt) + random.random()
                self.log(f"API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
        raise RuntimeError("Max retries exceeded for LLM call.")

    def extract_json(self, text):
        """Safely extract and parse JSON from LLM response text."""
        if not text:
            return None
        text = text.strip()
        
        # Try parsing directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try parsing code block
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Try finding the first '[' and last ']'
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
                
        # Try finding the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
                
        return None

    def count_words(self, text):
        """Count Chinese characters and English words."""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        other_chars = len(re.findall(r'[\u3000-\u303f\uff00-\uffef\d]+', text))
        return chinese_chars + english_words + other_chars

    def clean_section_headers(self, text, chapter_num=None, title=None):
        """Strip structural section headers, book titles, and redundant chapter headers from the text."""
        if not text:
            return text
            
        struct_pattern = re.compile(
            r'^\s*(#+\s*)?(第[一二三四五六七八九十]部分|第[0-9]+部分|Part\s*[0-9]+|场景[一二三四五]|第[一二三四五]幕|引子|场景|幕)\s*[:：\-\s\d]*.*$',
            re.IGNORECASE
        )
        
        h1_h2_pattern = re.compile(r'^\s*#{1,2}(\s+.*)?$')
        
        # Generic chapter header pattern (e.g., "第一章", "第1章", etc.)
        plain_chapter_pattern = re.compile(
            r'^\s*第\s*([0-9]+|[一二三四五六七八九十百千零]+)\s*[章回节幕说]((续|续一|续二|第二部分|第三部分)[\s\d]*)?[\s：:\-]*.*$',
            re.IGNORECASE
        )
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            s_line = line.strip()
            if not s_line:
                cleaned_lines.append(line)
                continue
                
            # 1. Strip structural patterns
            if struct_pattern.match(s_line):
                continue
                
            # 2. Strip H1/H2 markdown headings
            if h1_h2_pattern.match(s_line):
                continue
                
            # 3. Strip plain chapter titles if they match chapter num/title context
            is_redundant_header = False
            
            # If the line is short, check if it's a chapter header
            if len(s_line) < 50:
                if plain_chapter_pattern.match(s_line):
                    is_redundant_header = True
                elif title and title.strip("[]()（）【】") in s_line:
                    is_redundant_header = True
                elif chapter_num is not None and f"第{chapter_num}章" in s_line:
                    is_redundant_header = True
                    
            if is_redundant_header:
                continue
                
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)

    def initialize_tasks_queue(self):
        """Analyze outline and divide into a tasks queue chunk-by-chunk using LLM."""
        if os.path.exists(self.queue_file):
            self.log("Found existing tasks_queue.json. Skipping initialization.")
            return
            
        self.log(f"Initializing tasks_queue.json for '{self.title}' from startup.md...")
        
        if not os.path.exists(self.startup_file):
            raise FileNotFoundError(f"Startup outline file not found: {self.startup_file}")
            
        with open(self.startup_file, "r", encoding="utf-8") as f:
            outline_content = f.read()
            
        # Step A: Split outline into volumes
        num_volumes = max(1, self.total_chapters // 10)
        self.log(f"Dividing outline into {num_volumes} volumes...")
        
        split_prompt = f"""
你是一位金牌网络小说总编与大纲架构师。
我们正准备基于以下小说大纲进行长篇小说写作：
---
{outline_content}
---
这部小说计划共写作 {self.total_chapters} 章。
请将这个小说大纲合理划分为刚好 {num_volumes} 个逻辑分卷（Volume）。
要求：
1. 每一卷有卷序号、卷名、该卷的情节摘要（至少100字）。
2. 每卷必须有明确的起始章节号 `start_ch` 和结束章节号 `end_ch`。
3. 卷与卷的章节范围必须连续，覆盖第 1 章到第 {self.total_chapters} 章，且不能有重叠或缺失。

请严格返回一个 JSON 数组，包含刚好 {num_volumes} 个卷对象，格式如下：
[
  {{
    "idx": 卷序号(1到{num_volumes}的整数),
    "name": "分卷名称",
    "summary": "该卷的主要情节发展与冲突概述",
    "start_ch": 起始章节号(整数),
    "end_ch": 结束章节号(整数)
  }},
  ...
]

请确保只返回 JSON 数组，不要包含任何 markdown 代码块标记（如 ```json）或前后聊天解释。
"""
        
        volumes = None
        for attempt in range(4):
            try:
                res_text = self.call_llm(split_prompt, temperature=0.5)
                volumes = self.extract_json(res_text)
                if volumes and isinstance(volumes, list) and len(volumes) == num_volumes:
                    # Validate chapter ranges
                    valid_range = True
                    expected_ch = 1
                    for vol in sorted(volumes, key=lambda x: x.get("idx", 0)):
                        if vol.get("start_ch") != expected_ch or vol.get("end_ch") < vol.get("start_ch"):
                            valid_range = False
                            break
                        expected_ch = vol.get("end_ch") + 1
                    
                    if expected_ch - 1 == self.total_chapters and valid_range:
                        self.log(f"Successfully divided outline into {num_volumes} volumes.")
                        break
                self.log(f"Volume partitioning validation failed (attempt {attempt+1}/4). Retrying...")
            except Exception as e:
                self.log(f"Volume partitioning failed (attempt {attempt+1}/4): {e}")
            time.sleep(2)
            
        if not volumes:
            # Fallback to even chapter partition
            self.log("Warning: Volume partitioning using LLM failed. Using default equal partitions.")
            volumes = []
            ch_per_vol = self.total_chapters // num_volumes
            for idx in range(1, num_volumes + 1):
                start_ch = (idx - 1) * ch_per_vol + 1
                end_ch = idx * ch_per_vol if idx < num_volumes else self.total_chapters
                volumes.append({
                    "idx": idx,
                    "name": f"第{idx}卷：故事的铺陈与发展",
                    "summary": "故事继续在主线大纲框架下向前推进与演变。",
                    "start_ch": start_ch,
                    "end_ch": end_ch
                })
                
        # Step B: Generate chapter details chunk-by-chunk (5 chapters per chunk)
        all_chapters = []
        for vol in volumes:
            vol_name = vol["name"]
            vol_summary = vol["summary"]
            start_ch = vol["start_ch"]
            end_ch = vol["end_ch"]
            
            self.log(f"Planning chapters for Volume {vol['idx']} ({vol_name}), Chapters {start_ch} to {end_ch}...")
            
            chunk_size = 5
            for chunk_start in range(start_ch, end_ch + 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size - 1, end_ch)
                num_ch = chunk_end - chunk_start + 1
                
                self.log(f"   Planning Chapters {chunk_start} to {chunk_end}...")
                
                ch_prompt = f"""
你是一位金牌网络小说大纲策划。
我们正在写作小说《{self.title}》，风格属于：{self.style}。
当前正在规划：
- 卷名：{vol_name}
- 本卷梗概：{vol_summary}

整部小说的原始大纲：
---
{outline_content}
---

请为本分卷中的第 {chunk_start} 章到第 {chunk_end} 章（共 {num_ch} 章）规划详细的章节大纲。
每一章需要设计其：标题、核心矛盾冲突、出场人物、本章埋下的伏笔或呼应的伏笔、以及详细的剧情起承转合。

请严格返回一个 JSON 数组，包含刚好 {num_ch} 个章节对象，格式如下：
[
  {{
    "chapter_num": 章节序号(介于{chunk_start}和{chunk_end}之间的整数),
    "title": "章节标题",
    "volume": "{vol_name}",
    "conflict": "本章的核心矛盾冲突（详细描写，至少50字）",
    "characters": ["主要出场角色1", "主要出场角色2", ...],
    "foreshadowing": "本章需要埋下或呼应的伏笔/线索（详细描写，至少30字）",
    "summary": "本章的详细情节大纲与起承转合（详细描写，至少150字。写出具体的事件经过和解决办法）"
  }},
  ...
]

确保只返回 JSON 数组，不要包含 ```json 等任何额外包装。
"""
                
                success = False
                for attempt in range(4):
                    try:
                        res_text = self.call_llm(ch_prompt, temperature=0.6)
                        chapters = self.extract_json(res_text)
                        if chapters and isinstance(chapters, list) and len(chapters) == num_ch:
                            valid = True
                            for ch in chapters:
                                if not all(k in ch for k in ["chapter_num", "title", "volume", "conflict", "characters", "foreshadowing", "summary"]):
                                    valid = False
                                    break
                            if valid:
                                for ch in chapters:
                                    ch["status"] = "pending"
                                all_chapters.extend(chapters)
                                success = True
                                break
                        self.log(f"   Validation failed for chunk {chunk_start}-{chunk_end} (attempt {attempt+1}/4). Retrying...")
                    except Exception as e:
                        self.log(f"   Planning chunk failed: {e}")
                    time.sleep(2)
                    
                if not success:
                    # Make fallback chapter templates
                    self.log(f"   Warning: Failed to generate plan for Chapters {chunk_start}-{chunk_end}. Creating fallback chapters.")
                    for ch_num in range(chunk_start, chunk_end + 1):
                        all_chapters.append({
                            "chapter_num": ch_num,
                            "title": f"第{ch_num}章 命运的齿轮",
                            "volume": vol_name,
                            "conflict": "主角面临新的挑战与危机，必须在前行中寻找突破。",
                            "characters": ["主角"],
                            "foreshadowing": "本章在暗处埋下关于下一步计划的伏笔。",
                            "summary": "主角在这一章继续推进核心计划，通过一系列的探索和与配角的互动，解决阶段性的障碍，为下文做铺垫。",
                            "status": "pending"
                        })
                time.sleep(1)
                
        # Sort chapters
        all_chapters.sort(key=lambda x: x["chapter_num"])
        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump(all_chapters, f, ensure_ascii=False, indent=2)
            
        self.log(f"Successfully initialized tasks_queue.json with {len(all_chapters)} chapters.")

    def initialize_world_memory(self):
        """Analyze startup.md and extract initial world memory."""
        if os.path.exists(self.memory_file):
            self.log("Found existing world_memory.json. Skipping initialization.")
            return
            
        self.log("Initializing world_memory.json...")
        
        if not os.path.exists(self.startup_file):
            raise FileNotFoundError(f"Startup outline file not found: {self.startup_file}")
            
        with open(self.startup_file, "r", encoding="utf-8") as f:
            outline_content = f.read()
            
        prompt = f"""
根据以下小说大纲，分析并提取该小说在开篇时的初始世界记忆数据库：
---
{outline_content}
---

请提取以下内容：
1. `characters`: 主要出场的几位核心角色。包括他们的：status (当前状态，如隐居、在逃、在校等)、location (所在位置)、emotion (情感状态与态度)、personality (性格特征和目前的心智成长状态)、description (背景描述，约50字)。
2. `items`: 主要的核心道具、技术或装备。包括：location (位置/持有者)、status (状态描述)。
3. `relationships`: 核心角色之间的初始关系（如 顾远与林策 -> 对立与防备）。
4. `plot_threads`: 初始就存在的剧情线或需要解决悬念列表（列表格式）。

请严格返回一个 JSON 对象，结构如下：
{{
  "characters": {{
    "角色姓名": {{
      "status": "当前状态",
      "location": "所处位置",
      "emotion": "情感线状态",
      "personality": "性格特征及心智阶段状态",
      "description": "背景描述"
    }}
  }},
  "items": {{
    "道具/技术名": {{
      "location": "所有者/所处位置",
      "status": "状态/性能描述"
    }}
  }},
  "relationships": {{
    "人物A与人物B": "关系描述"
  }},
  "plot_threads": [
    "剧情线1/待解决悬念",
    "剧情线2"
  ],
  "completed_milestones": [
    "已发生的核心剧情里程碑/事件纪录"
  ]
}}

确保只返回 JSON 对象，不要包含 ```json 等任何额外包装。
"""
        
        memory_data = None
        for attempt in range(3):
            try:
                res_text = self.call_llm(prompt, temperature=0.5)
                memory_data = self.extract_json(res_text)
                if memory_data and isinstance(memory_data, dict) and "characters" in memory_data:
                    break
                self.log(f"Memory initialization validation failed (attempt {attempt+1}/3). Retrying...")
            except Exception as e:
                self.log(f"Memory initialization failed: {e}")
            time.sleep(2)
            
        if not memory_data:
            self.log("Warning: Failed to extract structured memory. Initializing with empty schema.")
            memory_data = {
                "characters": {},
                "items": {},
                "relationships": {},
                "plot_threads": ["推进故事主线，解开大纲中的核心悬念。"],
                "completed_milestones": []
            }
            
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
        self.log("Successfully initialized world_memory.json.")

    def get_recent_summaries(self, tasks_queue, current_idx, count=3):
        """Retrieve summaries of the most recently written chapters."""
        recent = []
        for i in range(current_idx - 1, max(-1, current_idx - 1 - count), -1):
            task = tasks_queue[i]
            if task.get("status") == "completed" and "written_summary" in task:
                recent.append(f"第 {task['chapter_num']} 章《{task['title']}》剧情摘要：{task['written_summary']}")
        recent.reverse()
        return "\n".join(recent) if recent else "这是小说的开篇第一章。"

    def write_chapter_in_parts(self, chapter_task, recent_context, memory, previous_chapter_ending=""):
        """Write the chapter in parts (scenes) to ensure length, depth and dialogue."""
        chapter_num = chapter_task["chapter_num"]
        title = chapter_task["title"]
        volume = chapter_task["volume"]
        conflict = chapter_task["conflict"]
        characters = ", ".join(chapter_task["characters"])
        foreshadowing = chapter_task["foreshadowing"]
        summary = chapter_task["summary"]
        
        ref_text_instruction = ""
        if self.ref_text:
            ref_text_instruction = f"\n【期望的写作风格与笔触样板】\n请模仿并匹配以下参考段落的语言风格、句式结构、节奏快慢和叙事张力进行创作：\n```\n{self.ref_text}\n```\n"

        prev_ending_instruction = ""
        if previous_chapter_ending:
            prev_ending_instruction = f"""
【前一章结尾原文参考】
以下是上一章（第 {chapter_num - 1} 章）结尾的最后几百字原文：
```
{previous_chapter_ending}
```
请务必做到：
- 完美衔接上一章结尾的场景、动作和人物状态。如果发生了时间或空间转移，请通过自然、流畅的文学描写进行“场景转换”（例如使用交代天气、环境变化、人物行进途中的过渡性叙述），绝对避免出现生硬的直接跳跃或场景错乱。
- 确保主角所处的具体物理位置（如庙宇、荒山、城市等）和正在发生的动作逻辑完全连贯。
"""

        self.log(f"   [Writing Part 1/3: Environment and Setup]...")
        prompt_p1 = f"""
你是一位文笔细腻、擅长铺陈描写的小说家。你正在写一部史诗长篇网文《{self.title}》，风格属于：{self.style}。

当前写作章节：
- 卷名：{volume}
- 章节：第 {chapter_num} 章
- 标题：{title}

本章大纲摘要：
{summary}
本章核心冲突：
{conflict}
出场角色：
{characters}
需要埋下的伏笔/细节：
{foreshadowing}

前情提要：
{recent_context}
{prev_ending_instruction}

世界记忆设定：
{json.dumps(memory, ensure_ascii=False, indent=2)}
{ref_text_instruction}
---
【写作规范】
1. 每一章的正文必须极其扎实、细节丰富。为了达到厚度，我们将分三部分撰写。现在你只需撰写【第一部分：场景引入与氛围铺垫】（约占全章篇幅的三分之一）。
2. 字数目标：1200 - 1500 字。
3. 要求：深度描写环境细节（如声音、光线、气味、材质质感），切忌概述性语言。展现角色出场时的内心活动与最初的张力。在写到三分之一处自然停下，留下后续故事发展的钩子。
4. 请直接输出正文，不要带章节名，不要有任何“第一部分开始”等解释性或批注文字。
5. 角色与情节一致性防线：
   - 必须严格遵循 `世界记忆设定` 中每个人物当前的最新“性格与心理成长状态（personality）”和“情感状态（emotion）”，绝不允许人物性格退化至前文已蜕变的旧阶段。
   - 仔细审查并对照 `completed_milestones`（已发生核心剧情里程碑），严禁重复描写或循环重现任何此前章节已发生的重大相遇、决战或细节桥段。若大纲事件与历史类似，须写出剧情层面的递进。
"""
        
        part_1 = self.call_llm(prompt_p1, temperature=self.temperature).strip()
        time.sleep(1)
        
        self.log(f"   [Writing Part 2/3: Core Conflict and Dialogue]...")
        prompt_p2 = f"""
你正在创作小说《{self.title}》（{self.style}）。刚刚写完第 {chapter_num} 章《{title}》的第一部分。

本章剧情大纲：
{summary}
核心冲突：
{conflict}

第一部分内容如下：
---
{part_1}
---
{ref_text_instruction}
---
【写作规范】
1. 接着第一部分的内容，继续撰写【第二部分：冲突爆发与多轮对话】（约占全章篇幅的三分之一）。
2. 字数目标：1200 - 1500 字。
3. 要求：保持与前文风格、语调和叙事节奏的绝对一致。在此部分，让本章的核心冲突正面碰撞，设计多轮具有潜台词、机锋和性格特色的人物对话。详细描写对话时的眼神、神态、肢体细节与内心算计。在写到三分之二处自然停下，留有未完悬念。
4. 请直接输出接续部分的正文，不要有任何解释性、标记性或批注文字。
5. 角色与情节一致性防线：严格遵循每个人物最新的心智成长状态和情感定位，拒绝任何与先前章节已发生剧情矛盾的行为退化，确保剧情连贯合理。
"""
        
        part_2 = self.call_llm(prompt_p2, temperature=self.temperature).strip()
        time.sleep(1)
        
        self.log(f"   [Writing Part 3/3: Resolution and Foreshadowing]...")
        prompt_p3 = f"""
你正在创作小说《{self.title}》（{self.style}）。已经写完第 {chapter_num} 章《{title}》的前面两个部分。

本章大纲：{summary}
核心冲突：{conflict}
本章要求埋下的伏笔：{foreshadowing}

前文内容如下：
---
{part_1}

{part_2}
---
{ref_text_instruction}
---
【写作规范】
1. 接着前文，撰写【第三部分：冲突收尾与高潮余韵】（完成本章）。
2. 字数目标：1200 - 1500 字。
3. 要求：解决或悬置第二部分的冲突，迎来本章的阶段性结果与高潮。一定要在细节处不露痕迹地埋下本章要求的伏笔：【{foreshadowing}】。直接以章节的自然叙述结束，输出的内容应能与前文无缝拼接。
4. 请直接输出接续部分的正文，不要有任何解释性、标记性或批注文字。
5. 角色与情节一致性防线：确保本章的收尾在角色心智状态和历史事件的连贯上无懈可击，坚决避免任何人物性格的退化或历史情节的重复发生。
"""
        
        part_3 = self.call_llm(prompt_p3, temperature=self.temperature).strip()
        
        full_text = f"{part_1}\n\n{part_2}\n\n{part_3}"
        return self.clean_section_headers(full_text, chapter_num, title)

    def check_and_expand_chapter(self, chapter_text, chapter_task):
        """Check the word count, and expand the chapter dynamically if below limit."""
        chapter_num = chapter_task.get("chapter_num")
        title = chapter_task.get("title")
        chapter_text = self.clean_section_headers(chapter_text, chapter_num, title)
        word_count = self.count_words(chapter_text)
        if word_count >= self.min_words:
            return chapter_text, word_count
            
        self.log(f"   ⚠️ Chapter {chapter_task['chapter_num']} word count is {word_count}, lower than target {self.min_words}. Initiating expansion loop...")
        
        ref_text_instruction = ""
        if self.ref_text:
            ref_text_instruction = f"\n【期望的写作风格与笔触样板】\n请模仿并匹配以下参考段落的语言风格、句式结构、节奏快慢和叙事张力进行创作：\n```\n{self.ref_text}\n```\n"

        attempts = 0
        while word_count < self.min_words and attempts < 3:
            attempts += 1
            self.log(f"   Expanding chapter (attempt {attempts}/3)...")
            
            prompt = f"""
你是一位文字细腻的文学家。正在为小说《{self.title}》创作【第 {chapter_task['chapter_num']} 章：{chapter_task['title']}】。
目前写的正文总字数仅有 {word_count} 字，低于我们 {self.min_words} 字的最低要求。

为了扩充篇幅，请在不改变原剧情逻辑和走势的前提下，对下面的内容进行深度细节扩写：
1. 深入描写人物在对话时的微表情、小动作、眼神动作，揭示潜台词。
2. 拓展环境描写，引入多重感官体验（视觉、触觉、嗅觉、甚至空气的震颤、细微的尘埃、材质的温度）。
3. 拓展人物的内心挣扎、对现状的权衡、或者联想到的回忆。
4. 扩写关键对决或对话细节，让节奏更有张力。
{ref_text_instruction}
已有的正文内容：
---
{chapter_text}
---

请重新输出经过深度扩写后的完整章节内容（目标字数在 {self.min_words} 到 {self.max_words} 之间），直接输出正文，不要包含任何旁白和标记：
"""
            try:
                chapter_text = self.call_llm(prompt, temperature=self.temperature + 0.05).strip()
                chapter_text = self.clean_section_headers(chapter_text, chapter_num, title)
                word_count = self.count_words(chapter_text)
                self.log(f"   Expansion attempt {attempts} finished. New word count: {word_count}.")
            except Exception as e:
                self.log(f"   Expansion call failed: {e}")
                
        return chapter_text, word_count

    def update_world_memory(self, chapter_num, title, chapter_text, current_memory):
        """Call LLM to extract latest updates (characters, items, relationships, plot threads) and merge them."""
        self.log(f"   [Extracting memory updates from Chapter {chapter_num}]...")
        
        prompt = f"""
你是一位小说世界观与剧情线记录官。
根据刚刚写完的第 {chapter_num} 章《{title}》的剧情，对比现有的世界记忆，提取出在本章中发生了状态/位置/情感/性格变化的人物、有去向或状态变动的道具、人际关系变化、新增或收回的悬念伏笔、以及本章完成的重要剧情里程碑。

当前的世界记忆：
{json.dumps(current_memory, ensure_ascii=False, indent=2)}

刚刚写完的章节正文（前 3500 字）：
---
{chapter_text[:3500]}
---

请提取本章发生的变化，并返回一个 JSON 对象。
注意：
1. 只输出发生了变动的人物或道具，如果完全没变，请不要在返回的 JSON 中包含该人物/道具。
2. 格式必须是：
{{
  "characters": {{
    "发生变化的人物姓名": {{
      "status": "本章最新的具体状态（详细描写，字数在80字左右）",
      "location": "本章结束时所处的具体位置",
      "emotion": "本章最新的情感线变化",
      "personality": "本章最新的性格特征与心理成长变化（若本章中其性格、心智、处事作风发生了改变或蜕变，请在此详细描述其最新状态，例如：已由先前的隐忍退让转变为果断无情；若没有变化则不填）",
      "description": "如果是首次出场，请提供背景描述，否则不要包含此字段"
    }}
  }},
  "items": {{
    "发生变化的道具/技术名称": {{
      "location": "本章最新所处位置或所有人",
      "status": "本章最新的道具状态描述"
    }}
  }},
  "relationships": {{
    "发生关系变化的人物（如 龙一与雷震）": "本章最新的关系变化"
  }},
  "plot_threads_added": ["本章新增的剧情线/悬念（如果没有则空数组）"],
  "plot_threads_removed": ["本章已解决或收起的剧情线/悬念（如果没有则空数组）"],
  "new_milestone": "本章完成的重要剧情里程碑描述（一句简明扼要的话，总结本章主角完成了什么实质遭遇、击败了谁或达成了什么目的，格式为：第X章：主角在XX做了XX；若无重大事件则不填或为空字符串）"
}}

请确保返回纯 JSON 对象，不要包含 ```json 等任何额外包装。
"""
        
        for attempt in range(3):
            try:
                res_text = self.call_llm(prompt, temperature=0.5)
                updates = self.extract_json(res_text)
                if updates and isinstance(updates, dict):
                    # Merge updates into current_memory manually
                    # Characters
                    if "characters" in updates and isinstance(updates["characters"], dict):
                        for char, data in updates["characters"].items():
                            if char not in current_memory["characters"]:
                                current_memory["characters"][char] = {}
                            for k, v in data.items():
                                current_memory["characters"][char][k] = v
                                
                    # Items
                    if "items" in updates and isinstance(updates["items"], dict):
                        for item, data in updates["items"].items():
                            if item not in current_memory["items"]:
                                current_memory["items"][item] = {}
                            for k, v in data.items():
                                current_memory["items"][item][k] = v
                                
                    # Relationships
                    if "relationships" in updates and isinstance(updates["relationships"], dict):
                        for rel, val in updates["relationships"].items():
                            current_memory["relationships"][rel] = val
                            
                    # Plot threads added
                    if "plot_threads_added" in updates and isinstance(updates["plot_threads_added"], list):
                        for thread in updates["plot_threads_added"]:
                            if thread not in current_memory["plot_threads"]:
                                current_memory["plot_threads"].append(thread)
                                
                    # Plot threads removed
                    if "plot_threads_removed" in updates and isinstance(updates["plot_threads_removed"], list):
                        for thread in updates["plot_threads_removed"]:
                            if thread in current_memory["plot_threads"]:
                                current_memory["plot_threads"].remove(thread)
                            else:
                                # Loose prefix match removal
                                for active_thread in list(current_memory["plot_threads"]):
                                    if thread in active_thread or active_thread in thread:
                                        current_memory["plot_threads"].remove(active_thread)
                                        break
                                        
                    # Completed Milestones
                    if "completed_milestones" not in current_memory:
                        current_memory["completed_milestones"] = []
                    new_milestone = updates.get("new_milestone")
                    if new_milestone and isinstance(new_milestone, str):
                        new_milestone = new_milestone.strip()
                        if new_milestone and new_milestone not in current_memory["completed_milestones"]:
                            current_memory["completed_milestones"].append(new_milestone)
                                        
                    # Save memory
                    with open(self.memory_file, "w", encoding="utf-8") as f:
                        json.dump(current_memory, f, ensure_ascii=False, indent=2)
                        
                    self.log(f"   World memory successfully updated and saved for Chapter {chapter_num}.")
                    return current_memory
                self.log(f"   Memory extraction validation failed (attempt {attempt+1}/3). Retrying...")
            except Exception as e:
                self.log(f"   Memory extraction failed: {e}")
            time.sleep(2)
            
        self.log("   Warning: Failed to extract updates. World memory remains unchanged.")
        return current_memory

    def run_loop(self, check_stop_callback=None):
        """Execute the novel writing pipeline loop. Calls check_stop_callback() to support pause."""
        self.check_stop_callback = check_stop_callback
        self.log(f"Starting novel generation loop for project: {self.title}")
        
        # 1. Initialize folders and files
        os.makedirs(os.path.join(self.project_path, "docs"), exist_ok=True)
        self.initialize_tasks_queue()
        self.initialize_world_memory()
        
        # Load tasks
        with open(self.queue_file, "r", encoding="utf-8") as f:
            tasks_queue = json.load(f)
            
        # Load memory
        with open(self.memory_file, "r", encoding="utf-8") as f:
            world_memory = json.load(f)
            
        # Initialize master novel
        if not os.path.exists(self.novel_file):
            with open(self.novel_file, "w", encoding="utf-8") as f:
                f.write(f"# 《{self.title}》\n\n> 风格特征：{self.style}\n> 总章节规划：{self.total_chapters} 章\n\n---\n\n")
                
        total_chapters = len(tasks_queue)
        completed_count = sum(1 for t in tasks_queue if t.get("status") == "completed")
        self.log(f"Initial progress: {completed_count}/{total_chapters} chapters completed.")
        
        for idx, task in enumerate(tasks_queue):
            if task.get("status") == "completed":
                continue
                
            # Check for stop request (pause or cancel)
            if check_stop_callback and check_stop_callback():
                self.log("Generation paused cooperatively by backend manager.")
                return False
                
            chapter_num = task["chapter_num"]
            title = task["title"]
            era = task.get("volume", "第一卷")
            conflict = task["conflict"]
            chars = ", ".join(task["characters"])
            foreshadowing = task["foreshadowing"]
            
            self.log(f"\n==================================================")
            self.log(f"🎬 WRITING CHAPTER: 第 {chapter_num} / {total_chapters} 章 《{title}》")
            self.log(f"📂 Volume: {era}")
            self.log(f"⚔️ Conflict: {conflict}")
            self.log(f"👥 Characters: {chars}")
            self.log(f"🔑 Foreshadowing: {foreshadowing}")
            self.log(f"==================================================")
            
            # Fetch recent context summaries
            recent_context = self.get_recent_summaries(tasks_queue, idx, count=3)
            
            # Fetch previous chapter ending text for seamless transitions
            previous_chapter_ending = ""
            if chapter_num > 1:
                prev_file = os.path.join(self.project_path, "docs", f"chapter_{chapter_num - 1}.txt")
                if os.path.exists(prev_file):
                    try:
                        with open(prev_file, "r", encoding="utf-8") as f:
                            prev_text = f.read().strip()
                        previous_chapter_ending = prev_text[-1000:]
                    except Exception as e:
                        self.log(f"   ⚠️ Failed to read previous chapter file: {e}")
                
                if not previous_chapter_ending and os.path.exists(self.novel_file):
                    try:
                        with open(self.novel_file, "r", encoding="utf-8") as f:
                            novel_content = f.read()
                        search_marker = f"## 第{chapter_num - 1}章"
                        if search_marker in novel_content:
                            parts = novel_content.split(search_marker)
                            if len(parts) > 1:
                                prev_chapter_part = parts[-1].split("---")[0].strip()
                                previous_chapter_ending = prev_chapter_part[-1000:]
                    except Exception as e:
                        self.log(f"   ⚠️ Failed to extract previous chapter ending from master_novel: {e}")
            
            try:
                # 1. Write initial text
                draft_text = self.write_chapter_in_parts(task, recent_context, world_memory, previous_chapter_ending=previous_chapter_ending)
                time.sleep(1)
                
                # 2. Expand if too short
                final_text, final_words = self.check_and_expand_chapter(draft_text, task)
                
                # 3. Append to master novel file
                with open(self.novel_file, "a", encoding="utf-8") as f:
                    clean_title = title.strip("[]()（）【】")
                    f.write(f"## 第{chapter_num}章 {clean_title}\n\n")
                    f.write(final_text)
                    f.write("\n\n---\n\n")
                self.log(f"💾 Chapter {chapter_num} written successfully to master_novel.md ({final_words} words).")
                
                # Also save to individual chapter file for seamless transition references
                try:
                    chapter_file = os.path.join(self.project_path, "docs", f"chapter_{chapter_num}.txt")
                    with open(chapter_file, "w", encoding="utf-8") as f:
                        f.write(final_text)
                except Exception as e:
                    self.log(f"   ⚠️ Failed to save individual chapter file: {e}")
                
                # 4. Update memory
                world_memory = self.update_world_memory(chapter_num, title, final_text, world_memory)
                time.sleep(1)
                
                # 5. Generate a short summary for tasks queue context
                self.log(f"   [Generating chapter summary for future context]...")
                summary_prompt = f"""
请为第 {chapter_num} 章《{title}》的正文生成一段 120 字以内的简要剧情摘要。
主要记录：本章发生了什么核心情节，有哪些人际关系或局势的实质性推进，以便为后续章节的写作提供准确的前情参考。

章节正文：
---
{final_text[:3000]}
---

请直接输出摘要，不要有任何标记或解释：
"""
                written_summary = self.call_llm(summary_prompt, temperature=0.5).strip()
                
                # 6. Update task queue
                task["status"] = "completed"
                task["written_summary"] = written_summary
                task["word_count"] = final_words
                task["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                with open(self.queue_file, "w", encoding="utf-8") as f:
                    json.dump(tasks_queue, f, ensure_ascii=False, indent=2)
                    
                completed_count += 1
                self.log(f"✅ Chapter {chapter_num} processed. Summary: {written_summary}")
                
                # Cool down
                time.sleep(1.5)
                
            except GenerationPausedException:
                self.log("Generation paused cooperatively by backend manager (mid-chapter).")
                task["status"] = "pending"
                with open(self.queue_file, "w", encoding="utf-8") as f:
                    json.dump(tasks_queue, f, ensure_ascii=False, indent=2)
                return False
            except Exception as e:
                self.log(f"❌ Error writing chapter {chapter_num}: {e}")
                self.log(traceback.format_exc())
                task["status"] = "error"
                task["error_msg"] = str(e)
                with open(self.queue_file, "w", encoding="utf-8") as f:
                    json.dump(tasks_queue, f, ensure_ascii=False, indent=2)
                return False
                
        self.log(f"🎉 Complete! All {total_chapters} chapters generated successfully.")
        return True
