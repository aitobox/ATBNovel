# -*- coding: utf-8 -*-

import os
import re
import json
import time
import shutil
import threading
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import our NovelGenerator
from generator import NovelGenerator

# Load env variables
load_dotenv()

def load_model_configs():
    configs = []
    seen_names = set()
    
    def add_configs(names_str, key, base):
        if not names_str or not key:
            return
        # Split by comma to support multiple models on the same channel/base
        names = [n.strip().strip("'\"") for n in names_str.split(",")]
        key = key.strip("'\"")
        base = base.strip("'\"") if base else None
        
        for name in names:
            if name and name not in seen_names:
                seen_names.add(name)
                configs.append({
                    "name": name,
                    "api_key": key,
                    "api_base": base
                })
            
    # Try default first
    default_name = os.environ.get("OPENAI_MODEL_NAME", "deepseek-v4-flash")
    default_key = os.environ.get("OPENAI_API_KEY")
    default_base = os.environ.get("OPENAI_API_BASE")
    if default_key:
        add_configs(default_name, default_key, default_base)
    
    # Scan for MODEL_N_NAME, MODEL_N_API_KEY, MODEL_N_API_BASE
    for i in range(1, 100):
        name = os.environ.get(f"MODEL_{i}_NAME")
        key = os.environ.get(f"MODEL_{i}_API_KEY")
        base = os.environ.get(f"MODEL_{i}_API_BASE")
        if name and key:
            add_configs(name, key, base)
            
    return configs


app = FastAPI(title="ATBNovel Web Management Platform")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "projects"))
if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR, exist_ok=True)

# Background generation task registry
active_generators = {}  # project_id -> { "thread": Thread, "stop_event": Event }
active_generators_lock = threading.Lock()

def get_project_path(project_id: str) -> str:
    """Strictly validate project_id and return resolved absolute path inside the sandbox."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID format. Alphanumeric, underscores, and dashes only.")
    
    # Resolve absolute path
    project_path = os.path.abspath(os.path.join(PROJECTS_DIR, project_id))
    # Security trailing slash boundary check to prevent partial matching bypasses
    if not project_path.startswith(PROJECTS_DIR + os.path.sep) and project_path != PROJECTS_DIR:
        raise HTTPException(status_code=400, detail="Path traversal attempt detected.")
        
    return project_path

def run_project_generation(project_id: str, project_path: str, config: dict):
    """Target function for background thread execution."""
    configs = load_model_configs()
    selected_model = config.get("model_name")
    
    matched_config = None
    if selected_model:
        for cfg in configs:
            if cfg["name"] == selected_model:
                matched_config = cfg
                break
                
    if not matched_config and configs:
        matched_config = configs[0]
        
    if not matched_config:
        # Fallback to general environment variables if load_model_configs didn't produce anything
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE")
        model_name = os.environ.get("OPENAI_MODEL_NAME", "deepseek-v4-flash")
        if model_name:
            model_name = [n.strip().strip("'\"") for n in model_name.split(",")][0]
        else:
            model_name = "deepseek-v4-flash"
        if api_key:
            matched_config = {
                "name": model_name,
                "api_key": api_key.strip("'\""),
                "api_base": api_base.strip("'\"") if api_base else None
            }
            
    if not matched_config:
        log_file = os.path.join(project_path, "generation.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ FATAL: No valid model configuration found.\n")
        return

    api_key = matched_config["api_key"]
    api_base = matched_config["api_base"]
    model_name = matched_config["name"]

    # Instantiate generator
    generator = NovelGenerator(
        project_path=project_path,
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        config=config
    )
    
    def check_stop():
        with active_generators_lock:
            state = active_generators.get(project_id)
            if state and state["stop_event"].is_set():
                return True
        return False

    generator.log("🚀 Background novel generation pipeline started.")
    try:
        success = generator.run_loop(check_stop_callback=check_stop)
        if success:
            generator.log("🎉 Background novel generation completed successfully.")
        else:
            generator.log("⏸️ Background novel generation paused or interrupted.")
    except Exception as e:
        generator.log(f"💥 Background novel generation failed with fatal error: {e}")
        generator.log(traceback.format_exc())
    finally:
        with active_generators_lock:
            if project_id in active_generators:
                del active_generators[project_id]

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the main frontend application file."""
    template_path = os.path.abspath(os.path.join(BASE_DIR, "templates", "index.html"))
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Index template not found.")
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/api/config")
def get_global_config():
    """Expose non-sensitive configuration parameters to the client dashboard."""
    configs = load_model_configs()
    model_names = [cfg["name"] for cfg in configs]
    return {
        "model_names": model_names,
        "model_name": model_names[0] if model_names else "deepseek-v4-flash"
    }

@app.get("/api/projects")
def list_projects():
    """Scan the sandbox directory and fetch a summary of all projects."""
    projects = []
    for name in os.listdir(PROJECTS_DIR):
        project_path = os.path.join(PROJECTS_DIR, name)
        if os.path.isdir(project_path):
            config_file = os.path.join(project_path, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    
                    # Compute progress stats
                    queue_file = os.path.join(project_path, "tasks_queue.json")
                    completed = 0
                    total = config.get("total_chapters", 100)
                    progress_status = "pending"
                    
                    if os.path.exists(queue_file):
                        with open(queue_file, "r", encoding="utf-8") as q_f:
                            queue = json.load(q_f)
                            total = len(queue)
                            completed = sum(1 for ch in queue if ch.get("status") == "completed")
                            if completed == total and total > 0:
                                progress_status = "completed"
                            elif completed > 0:
                                progress_status = "running"
                                
                    with active_generators_lock:
                        is_active = name in active_generators
                        
                    projects.append({
                        "id": name,
                        "title": config.get("title", name),
                        "style": config.get("style", "科幻"),
                        "total_chapters": total,
                        "progress": completed,
                        "is_active": is_active,
                        "status": "active" if is_active else progress_status,
                        "created_at": config.get("created_at", "未知")
                    })
                except Exception as e:
                    # Log silently to server stdout
                    print(f"Error loading project directory '{name}': {e}")
    
    # Sort by created time or name descending
    projects.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return projects

@app.post("/api/projects")
def create_project(
    title: str = Body(...),
    outline: str = Body(...),
    total_chapters: int = Body(100),
    min_words: int = Body(3000),
    max_words: int = Body(5000),
    style: str = Body("科幻"),
    model_name: Optional[str] = Body(None),
    ref_text: Optional[str] = Body(None)
):
    """Receive configurations and outline and initialize folders."""
    # Generate clean ID slug using ASCII alphanumeric, underscores, and dashes only
    clean_title = re.sub(r'[^a-zA-Z0-9_\-]', '', title)
    clean_title = clean_title.strip('_').strip('-')
    clean_title = clean_title[:25]
    if not clean_title:
        clean_title = "novel"
    project_id = f"{clean_title}_{int(time.time())}"
    
    project_path = get_project_path(project_id)
    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, "docs"), exist_ok=True)
    
    # Save startup.md
    with open(os.path.join(project_path, "docs", "startup.md"), "w", encoding="utf-8") as f:
        f.write(outline)
        
    # Save config.json
    config = {
        "title": title,
        "total_chapters": total_chapters,
        "min_words": min_words,
        "max_words": max_words,
        "style": style,
        "model_name": model_name,
        "ref_text": ref_text,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(project_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        
    return {"status": "created", "project_id": project_id, "config": config}

@app.post("/api/projects/{project_id}/config")
def update_project_config(
    project_id: str,
    title: str = Body(...),
    outline: str = Body(...),
    total_chapters: int = Body(100),
    min_words: int = Body(3000),
    max_words: int = Body(5000),
    style: str = Body("科幻"),
    model_name: Optional[str] = Body(None),
    reset_progress: bool = Body(False),
    ref_text: Optional[str] = Body(None)
):
    """Update project config and outline mid-way, with optional progress reset."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found.")
        
    with active_generators_lock:
        if project_id in active_generators:
            raise HTTPException(status_code=400, detail="Cannot edit configuration of a running project. Please pause it first.")
            
    # Load old config if exists to keep created_at
    config_file = os.path.join(project_path, "config.json")
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                old_config = json.load(f)
                created_at = old_config.get("created_at", created_at)
        except Exception:
            pass
            
    # Save outline
    with open(os.path.join(project_path, "docs", "startup.md"), "w", encoding="utf-8") as f:
        f.write(outline)
        
    # Save new config
    config = {
        "title": title,
        "total_chapters": total_chapters,
        "min_words": min_words,
        "max_words": max_words,
        "style": style,
        "model_name": model_name,
        "ref_text": ref_text,
        "created_at": created_at,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        
    if reset_progress:
        # Purge generated queue, memory, novel files
        files_to_remove = [
            os.path.join(project_path, "tasks_queue.json"),
            os.path.join(project_path, "world_memory.json"),
            os.path.join(project_path, "master_novel.md")
        ]
        for path in files_to_remove:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing {path}: {e}")
                    
        # Truncate log
        log_file = os.path.join(project_path, "generation.log")
        if os.path.exists(log_file):
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Project settings updated and progress reset.\n")
            except Exception:
                pass
                
    return {"status": "updated", "config": config}

@app.get("/api/projects/{project_id}")
def get_project_detail(project_id: str):
    """Retrieve detailed state: configuration, progress, tasks list, and memory database."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found.")
        
    # Load config
    config_file = os.path.join(project_path, "config.json")
    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            
    # Load tasks queue
    queue_file = os.path.join(project_path, "tasks_queue.json")
    tasks = []
    if os.path.exists(queue_file):
        try:
            with open(queue_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
        except Exception:
            pass
            
    # Load memory
    memory_file = os.path.join(project_path, "world_memory.json")
    memory = {}
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            pass
            
    # Load outline
    outline_file = os.path.join(project_path, "docs", "startup.md")
    outline = ""
    if os.path.exists(outline_file):
        try:
            with open(outline_file, "r", encoding="utf-8") as f:
                outline = f.read()
        except Exception:
            pass

    with active_generators_lock:
        is_active = project_id in active_generators
        
    return {
        "id": project_id,
        "config": config,
        "outline": outline,
        "tasks": tasks,
        "memory": memory,
        "is_active": is_active,
        "progress": sum(1 for t in tasks if t.get("status") == "completed"),
        "total": len(tasks)
    }

@app.post("/api/projects/{project_id}/control")
def control_project(project_id: str, action: str = Body(embed=True)):
    """Start, Pause, or Resume generation."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found.")
        
    # Read config
    config_file = os.path.join(project_path, "config.json")
    if not os.path.exists(config_file):
        raise HTTPException(status_code=500, detail="Project configuration file missing.")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    with active_generators_lock:
        if action in ["start", "resume"]:
            if project_id in active_generators:
                return {"status": "already_running"}
                
            stop_event = threading.Event()
            thread = threading.Thread(
                target=run_project_generation,
                args=(project_id, project_path, config),
                daemon=True
            )
            active_generators[project_id] = {
                "thread": thread,
                "stop_event": stop_event
            }
            thread.start()
            return {"status": "started"}
            
        elif action == "pause":
            if project_id not in active_generators:
                return {"status": "not_running"}
                
            active_generators[project_id]["stop_event"].set()
            return {"status": "pausing"}
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

@app.get("/api/projects/{project_id}/logs")
def get_logs(project_id: str, limit: int = 150):
    """Tail log entries for visual terminal reporting in UI."""
    project_path = get_project_path(project_id)
    log_file = os.path.join(project_path, "generation.log")
    
    if not os.path.exists(log_file):
        return {"logs": "Initializing generator...\nWaiting for process to output log streams...\n"}
        
    try:
        # Read the last N lines
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            tail = lines[-limit:] if len(lines) > limit else lines
            return {"logs": "".join(tail)}
    except Exception as e:
        return {"logs": f"Error loading logs: {e}"}

@app.get("/api/projects/{project_id}/novel")
def get_novel_content(project_id: str):
    """Fetch complete draft text or download."""
    project_path = get_project_path(project_id)
    novel_file = os.path.join(project_path, "master_novel.md")
    
    if not os.path.exists(novel_file):
        return {"content": "Novel draft is currently empty or has not started writing chapters yet."}
        
    try:
        with open(novel_file, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read master novel file: {e}")

@app.get("/api/projects/{project_id}/export")
def export_novel_file(project_id: str):
    """Serve master_novel.md as a direct file download attachment."""
    project_path = get_project_path(project_id)
    novel_file = os.path.join(project_path, "master_novel.md")
    
    if not os.path.exists(novel_file):
        raise HTTPException(status_code=404, detail="Novel manuscript file not generated yet.")
        
    # Get the title from config.json to create a clean filename
    config_file = os.path.join(project_path, "config.json")
    filename = f"{project_id}.md"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                title = config.get("title", project_id)
                # Keep alphanumeric, Chinese chars, underscores, and dashes
                clean_filename = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fff]', '', title)
                clean_filename = clean_filename.strip('_').strip('-')
                if clean_filename:
                    filename = f"{clean_filename}.md"
        except Exception:
            pass
            
    import urllib.parse
    ascii_filename = f"{project_id}.md"
    encoded_filename = urllib.parse.quote(filename)
    
    # Serve with secure headers, including RFC 5987 filename fallback for UTF-8
    headers = {
        "Content-Disposition": f"attachment; filename=\"{ascii_filename}\"; filename*=utf-8''{encoded_filename}",
        "X-Content-Type-Options": "nosniff"
    }
    return FileResponse(
        path=novel_file,
        media_type="text/markdown",
        headers=headers
    )

@app.post("/api/projects/{project_id}/delete")
def delete_project(project_id: str):
    """Delete a project and purge its files after validating it is not currently writing."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found.")
        
    with active_generators_lock:
        if project_id in active_generators:
            raise HTTPException(status_code=400, detail="Cannot delete a running project. Please pause it first.")
            
    try:
        shutil.rmtree(project_path)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project directory: {e}")
