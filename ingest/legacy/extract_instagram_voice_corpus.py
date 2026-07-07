#!/usr/bin/env python3
"""
Instagram Data Export - Voice Corpus Extractor
Extracts YOUR text content from Instagram data export.

Usage:
    1. Unzip both Instagram exports into one folder
    2. Run: python extract_instagram_voice_corpus.py ~/Downloads/instagram-export
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
YOUR_NAME = "Mitchell Williams"
YOUR_USERNAMES = ["_mitwilli", "mitwilli", "mitchellwilliams"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def decode_instagram_text(text):
    """Fix Instagram's encoding issues."""
    if not text:
        return ""
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

def unix_to_date(ts):
    """Convert unix timestamp to ISO date."""
    try:
        return datetime.fromtimestamp(ts).isoformat()
    except:
        return None

def unix_to_year(ts):
    """Convert unix timestamp to year."""
    try:
        return datetime.fromtimestamp(ts).year
    except:
        return None

def load_json_file(filepath):
    """Safely load a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"    Warning: Could not load {filepath}: {e}")
        return None

def find_json_files(base_path, pattern):
    """Find all JSON files matching a pattern."""
    return list(base_path.rglob(pattern))

# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

def extract_posts(base_path):
    """Extract post captions."""
    posts = []
    
    # Try various possible paths
    possible_paths = [
        base_path / "content" / "posts_1.json",
        base_path / "your_instagram_activity" / "content" / "posts_1.json",
        base_path / "media" / "posts" / "posts_1.json",
    ]
    
    # Also find any posts*.json files
    for json_file in base_path.rglob("posts*.json"):
        if json_file not in possible_paths:
            possible_paths.append(json_file)
    
    for posts_path in possible_paths:
        if not posts_path.exists():
            continue
        
        data = load_json_file(posts_path)
        if not data:
            continue
        
        # Handle different formats
        post_list = data if isinstance(data, list) else data.get("ig_posts", data.get("posts", []))
        
        for post in post_list:
            # Try to find caption in various locations
            caption = ""
            timestamp = None
            
            # Format 1: media array with title
            if "media" in post:
                for media_item in post.get("media", []):
                    if "title" in media_item:
                        caption = decode_instagram_text(media_item["title"])
                        timestamp = media_item.get("creation_timestamp")
                        break
            
            # Format 2: direct title
            if not caption and "title" in post:
                caption = decode_instagram_text(post["title"])
                timestamp = post.get("creation_timestamp", post.get("taken_at"))
            
            # Format 3: caption field
            if not caption and "caption" in post:
                caption = decode_instagram_text(post["caption"])
                timestamp = post.get("creation_timestamp", post.get("taken_at"))
            
            if not caption:
                continue
            
            posts.append({
                "source": "instagram",
                "type": "post_caption",
                "date": unix_to_date(timestamp),
                "year": unix_to_year(timestamp),
                "content": caption,
                "word_count": len(caption.split()),
            })
    
    print(f"    Found {len(posts):,} post captions")
    return posts

def extract_stories(base_path):
    """Extract story text/captions."""
    stories = []
    
    possible_paths = [
        base_path / "content" / "stories.json",
        base_path / "your_instagram_activity" / "content" / "stories.json",
    ]
    
    for stories_path in possible_paths:
        if not stories_path.exists():
            continue
        
        data = load_json_file(stories_path)
        if not data:
            continue
        
        story_list = data if isinstance(data, list) else data.get("ig_stories", [])
        
        for story in story_list:
            caption = decode_instagram_text(story.get("title", ""))
            if not caption:
                continue
            
            timestamp = story.get("creation_timestamp")
            
            stories.append({
                "source": "instagram",
                "type": "story",
                "date": unix_to_date(timestamp),
                "year": unix_to_year(timestamp),
                "content": caption,
                "word_count": len(caption.split()),
            })
    
    if stories:
        print(f"    Found {len(stories):,} story captions")
    return stories

def extract_comments(base_path):
    """Extract comments you've made on posts."""
    comments = []
    
    possible_paths = [
        base_path / "comments" / "post_comments.json",
        base_path / "your_instagram_activity" / "comments" / "post_comments_1.json",
        base_path / "comments" / "post_comments_1.json",
    ]
    
    # Find all comment files
    for json_file in base_path.rglob("*comment*.json"):
        if json_file not in possible_paths:
            possible_paths.append(json_file)
    
    for comments_path in possible_paths:
        if not comments_path.exists():
            continue
        
        data = load_json_file(comments_path)
        if not data:
            continue
        
        # Handle different formats
        if isinstance(data, dict):
            comment_list = data.get("comments_media_comments", 
                          data.get("post_comments", 
                          data.get("comments", [])))
        else:
            comment_list = data
        
        for comment in comment_list:
            content = ""
            timestamp = None
            
            # Format 1: string_list_data
            if "string_list_data" in comment:
                for item in comment["string_list_data"]:
                    content = decode_instagram_text(item.get("value", ""))
                    timestamp = item.get("timestamp")
                    break
            
            # Format 2: direct comment field
            if not content and "comment" in comment:
                content = decode_instagram_text(comment["comment"])
                timestamp = comment.get("timestamp")
            
            # Format 3: text field
            if not content and "text" in comment:
                content = decode_instagram_text(comment["text"])
                timestamp = comment.get("timestamp", comment.get("created_at"))
            
            if not content:
                continue
            
            comments.append({
                "source": "instagram",
                "type": "comment",
                "date": unix_to_date(timestamp),
                "year": unix_to_year(timestamp),
                "content": content,
                "word_count": len(content.split()),
            })
    
    print(f"    Found {len(comments):,} comments")
    return comments

def extract_messages(base_path):
    """Extract DM messages you've sent."""
    messages = []
    
    # Find message folders
    inbox_paths = [
        base_path / "messages" / "inbox",
        base_path / "your_instagram_activity" / "messages" / "inbox",
    ]
    
    for inbox_path in inbox_paths:
        if not inbox_path.exists():
            continue
        
        conversation_count = 0
        your_message_count = 0
        
        # Each conversation is a folder
        for conv_folder in inbox_path.iterdir():
            if not conv_folder.is_dir():
                continue
            
            conversation_count += 1
            
            # Find message JSON files
            for json_file in conv_folder.glob("message_*.json"):
                data = load_json_file(json_file)
                if not data or "messages" not in data:
                    continue
                
                participants = [p.get("name", "") for p in data.get("participants", [])]
                
                for msg in data["messages"]:
                    sender = msg.get("sender_name", "")
                    
                    # Check if this is YOUR message
                    sender_lower = sender.lower()
                    is_yours = any(uname.lower() in sender_lower for uname in YOUR_USERNAMES)
                    is_yours = is_yours or YOUR_NAME.lower() in sender_lower
                    
                    if not is_yours:
                        continue
                    
                    content = decode_instagram_text(msg.get("content", ""))
                    
                    # Skip empty, reactions, shares, etc.
                    if not content:
                        continue
                    if "shared a" in content.lower() and ("story" in content.lower() or "post" in content.lower() or "reel" in content.lower()):
                        continue
                    if content.startswith("Liked a message"):
                        continue
                    
                    your_message_count += 1
                    timestamp = msg.get("timestamp_ms")
                    
                    messages.append({
                        "source": "instagram_dm",
                        "type": "message",
                        "date": datetime.fromtimestamp(timestamp / 1000).isoformat() if timestamp else None,
                        "year": datetime.fromtimestamp(timestamp / 1000).year if timestamp else None,
                        "conversation": conv_folder.name,
                        "participants": participants,
                        "content": content,
                        "word_count": len(content.split()),
                    })
        
        if your_message_count > 0:
            print(f"    Found {your_message_count:,} of your DMs across {conversation_count:,} conversations")
    
    return messages

def extract_reels(base_path):
    """Extract reel captions."""
    reels = []
    
    possible_paths = [
        base_path / "content" / "reels.json",
        base_path / "your_instagram_activity" / "content" / "reels.json",
    ]
    
    for reels_path in possible_paths:
        if not reels_path.exists():
            continue
        
        data = load_json_file(reels_path)
        if not data:
            continue
        
        reel_list = data if isinstance(data, list) else data.get("ig_reels_media", [])
        
        for reel in reel_list:
            # Try media array first
            caption = ""
            timestamp = None
            
            if "media" in reel:
                for media_item in reel.get("media", []):
                    if "title" in media_item:
                        caption = decode_instagram_text(media_item["title"])
                        timestamp = media_item.get("creation_timestamp")
                        break
            
            if not caption:
                caption = decode_instagram_text(reel.get("title", ""))
                timestamp = reel.get("creation_timestamp")
            
            if not caption:
                continue
            
            reels.append({
                "source": "instagram",
                "type": "reel_caption",
                "date": unix_to_date(timestamp),
                "year": unix_to_year(timestamp),
                "content": caption,
                "word_count": len(caption.split()),
            })
    
    if reels:
        print(f"    Found {len(reels):,} reel captions")
    return reels

# ============================================================
# MAIN
# ============================================================

def main(export_path):
    base_path = Path(export_path)
    
    if not base_path.exists():
        print(f"Error: Path does not exist: {export_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("INSTAGRAM DATA EXPORT - VOICE CORPUS EXTRACTOR")
    print(f"{'='*60}")
    print(f"\nProcessing: {base_path}")
    
    all_content = []
    
    print("\n[1/5] Extracting post captions...")
    all_content.extend(extract_posts(base_path))
    
    print("\n[2/5] Extracting story captions...")
    all_content.extend(extract_stories(base_path))
    
    print("\n[3/5] Extracting reel captions...")
    all_content.extend(extract_reels(base_path))
    
    print("\n[4/5] Extracting comments...")
    all_content.extend(extract_comments(base_path))
    
    print("\n[5/5] Extracting DM messages...")
    all_content.extend(extract_messages(base_path))
    
    # Sort by date
    all_content.sort(key=lambda x: x.get("date") or "0000")
    
    # Statistics
    total_words = sum(item.get("word_count", 0) for item in all_content)
    by_source = defaultdict(int)
    by_type = defaultdict(int)
    by_year = defaultdict(int)
    
    for item in all_content:
        by_source[item["source"]] += 1
        by_type[item["type"]] += 1
        if item.get("year"):
            by_year[item["year"]] += 1
    
    output = {
        "extraction_info": {
            "source": "Instagram Data Export",
            "extracted_for": YOUR_NAME,
            "extracted_at": datetime.now().isoformat(),
            "original_path": str(base_path),
            "script_version": "1.0"
        },
        "statistics": {
            "total_items": len(all_content),
            "total_words": total_words,
            "by_source": dict(by_source),
            "by_type": dict(by_type),
            "by_year": dict(sorted(by_year.items())),
            "date_range": {
                "earliest": all_content[0].get("date") if all_content else None,
                "latest": all_content[-1].get("date") if all_content else None,
            }
        },
        "corpus": all_content,
    }
    
    # Write output
    output_filename = f"instagram_voice_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = base_path.parent / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal items: {len(all_content):,}")
    print(f"Total words: {total_words:,}")
    print(f"\nBy type:")
    for type_, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {type_}: {count:,}")
    
    if by_year:
        print(f"\nYear range: {min(by_year.keys())} - {max(by_year.keys())}")
    
    output_size = output_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"OUTPUT: {output_path}")
    print(f"SIZE: {output_size / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    if output_size > 30 * 1024 * 1024:
        print("\n⚠️  File is over 30MB. You may need to split it.")
        print("    Run: python split_meta_corpus.py <output_file>")
    else:
        print("\n✓ File is under 30MB - ready to upload!")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_instagram_voice_corpus.py ~/Downloads/instagram-export")
        print("\nSteps:")
        print("  1. Unzip your Instagram export(s)")
        print("  2. If you have two exports, unzip both into the same folder")
        print("  3. Run this script pointing to that folder")
        sys.exit(1)
    
    main(sys.argv[1])
