# ELOG Read-Only API Reference for RAG Systems

## Connection and Basic Setup

```python
import elog

# Connect to ELOG (read-only operations)
logbook = elog.open('https://elog-gfa.psi.ch/SwissFEL+commissioning/')

# With authentication if required
logbook = elog.open(
    'https://elog-gfa.psi.ch/SwissFEL+commissioning/',
    user='username',
    password='password'
)
```

## Core Reading Operations

### Get Message IDs
```python
# Get all message IDs (newest first)
message_ids = logbook.get_message_ids()          # Returns list of all message IDs
last_id = logbook.get_last_message_id()          # Returns most recent message ID

# Example: Get recent entries
recent_ids = message_ids[:10]                    # Last 10 entries
```

### Read Individual Messages
```python
# Read a specific message - returns tuple of (message, attributes, attachments)
message, attributes, attachments = logbook.read(message_id)

# Access message components:
# message: str - HTML or plain text content
# attributes: dict - Entry metadata and attributes  
# attachments: list - URLs to attached files
```

### Message Attributes Available
Based on SwissFEL ELOG, available attributes include:
- `$@MID@$`: Message ID  
- `Date`: Entry date (e.g., 'Mon, 18 Aug 2025 16:44:41 +0200')
- `When`: Unix timestamp
- `Author`: Entry author
- `Category`: Entry category ('Info', 'Problem Report', 'Routine', etc.)
- `System`: Machine system ('Laser', 'Undulator', 'Diagnostics', etc.)
- `Title`: Entry title
- `Domain`: Domain area
- `Section`: Machine section
- `Beamline`: Beamline identifier
- `Effect`: Effect description
- `Encoding`: Content encoding ('HTML', 'plain')

## Search Operations

### Basic Text Search
```python
# Simple text search
results = logbook.search("beam energy")                    # Exact phrase
results = logbook.search("undulator", n_results=10)        # Limit results
results = logbook.search("laser", scope="subtext")         # Search in message text only
results = logbook.search("error", scope="attribname")      # Search in attributes
results = logbook.search("beam", scope="all")              # Search everywhere
```

### Regular Expression Search
```python
# Multi-word regex patterns (term.*term syntax)
results = logbook.search("beam.*energy")                   # Beam AND energy anywhere
results = logbook.search("laser.*power")                   # Laser AND power
results = logbook.search("error.*undulator")               # Error AND undulator
results = logbook.search("current.*voltage")               # Current AND voltage

# Advanced multi-term patterns
results = logbook.search("beam.*energy.*6.0")              # All three terms
results = logbook.search("laser.*power.*setting")          # Laser power settings
```

### Attribute-Based Filtering
```python
# Filter by single attribute
results = logbook.search({'Author': 'Smith'})
results = logbook.search({'Category': 'Problem Report'})
results = logbook.search({'System': 'Laser'})

# Multiple attribute filters (AND logic)
results = logbook.search({
    'System': 'Laser',
    'Category': 'Seed laser operation'
})
```

## Data Extraction and Processing

### Extract Entry Data
```python
# Parse entry components
message, attributes, attachments = logbook.read(message_id)

# Extract key information
entry_id = attributes.get('$@MID@$')
title = attributes.get('Title', '')
author = attributes.get('Author', '')
date = attributes.get('Date', '')
category = attributes.get('Category', '')
system = attributes.get('System', '')
text_content = message  # May contain HTML

# Clean HTML content (if needed)
import re
clean_text = re.sub(r'<[^>]+>', '', message)  # Remove HTML tags
```

### Bulk Data Retrieval
```python
def get_recent_entries(logbook, days=7, max_entries=50):
    """Get recent entries for RAG context."""
    message_ids = logbook.get_message_ids()[:max_entries]
    entries = []
    
    for msg_id in message_ids:
        try:
            message, attrs, attachments = logbook.read(msg_id)
            entries.append({
                'id': msg_id,
                'title': attrs.get('Title', ''),
                'author': attrs.get('Author', ''),
                'date': attrs.get('Date', ''),
                'category': attrs.get('Category', ''),
                'system': attrs.get('System', ''),
                'text': message,
                'url': f"{logbook._url}{msg_id}"
            })
        except Exception as e:
            continue
    
    return entries

def search_entries_for_rag(logbook, query, max_results=20):
    """Search entries and format for RAG context."""
    # Try regex search first
    message_ids = logbook.search(query, n_results=max_results)
    
    entries = []
    for msg_id in message_ids:
        try:
            message, attrs, attachments = logbook.read(msg_id)
            entries.append({
                'id': msg_id,
                'title': attrs.get('Title', ''),
                'author': attrs.get('Author', ''),
                'date': attrs.get('Date', ''),
                'category': attrs.get('Category', ''),
                'system': attrs.get('System', ''),
                'text': re.sub(r'<[^>]+>', '', message),  # Clean HTML
                'url': f"{logbook._url}{msg_id}",
                'relevance_score': 1.0  # Can be adjusted based on search type
            })
        except Exception:
            continue
    
    return entries
```

## RAG-Optimized Search Patterns

### Technical Component Searches
```python
# Equipment-specific searches
laser_entries = logbook.search({'System': 'Laser'})
undulator_entries = logbook.search("undulator.*gap")
beam_diagnostics = logbook.search("BPM.*readback")

# Problem/Issue searches  
laser_problems = logbook.search("laser.*problem")
beam_losses = logbook.search("beam.*lost")
fault_reports = logbook.search({'Category': 'Problem Report'})
```

### Measurement and Parameter Searches
```python
# Energy measurements
energy_6gev = logbook.search("energy.*6.0")
energy_settings = logbook.search("beam.*energy")

# Current measurements  
current_200ua = logbook.search("current.*200")
current_measurements = logbook.search("current.*μA")

# Voltage and power
voltage_settings = logbook.search("voltage.*kV")
power_measurements = logbook.search("power.*MW")
```

### Time-Based and Operational Searches
```python
# Startup/shutdown procedures
startup_procedures = logbook.search("startup.*sequence")
morning_operations = logbook.search("morning.*beam")
shift_handovers = logbook.search("shift.*handover")

# Recent activities
today_entries = logbook.search("today.*beam")
recent_problems = logbook.search({'Category': 'Problem Report'})
```

## Error Handling and Performance

### Robust Search Functions
```python
def safe_search(logbook, query, max_results=20, timeout=30):
    """Safe search with error handling."""
    try:
        return logbook.search(query, n_results=max_results, timeout=timeout)
    except Exception as e:
        print(f"Search failed for '{query}': {e}")
        return []

def safe_read(logbook, message_id, timeout=30):
    """Safe read with error handling."""
    try:
        return logbook.read(message_id, timeout=timeout)
    except Exception as e:
        print(f"Read failed for message {message_id}: {e}")
        return None, {}, []
```

### Multi-Pattern Search for RAG
```python
def comprehensive_search(logbook, primary_query, related_terms=None, max_total=50):
    """Comprehensive search combining multiple strategies for RAG."""
    all_results = []
    
    # Primary search
    primary_results = safe_search(logbook, primary_query, max_results=max_total//2)
    all_results.extend(primary_results)
    
    # Related term searches
    if related_terms:
        remaining = max_total - len(all_results)
        per_term = max(1, remaining // len(related_terms))
        
        for term in related_terms:
            if len(all_results) >= max_total:
                break
            term_results = safe_search(logbook, term, max_results=per_term)
            all_results.extend(term_results)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for result in all_results:
        if result not in seen:
            unique_results.append(result)
            seen.add(result)
    
    return unique_results[:max_total]
```

## RAG Context Formatting

### Format Entries for RAG Context
```python
def format_entries_for_rag(entries, max_chars_per_entry=500):
    """Format ELOG entries for RAG context input."""
    formatted_entries = []
    
    for entry in entries:
        # Clean and truncate text
        clean_text = re.sub(r'<[^>]+>', '', entry['text'])
        if len(clean_text) > max_chars_per_entry:
            clean_text = clean_text[:max_chars_per_entry] + "..."
        
        # Create structured context
        context = f"""
ELOG Entry #{entry['id']} - {entry['title']}
Author: {entry['author']}
Date: {entry['date']}
Category: {entry['category']}
System: {entry['system']}
Content: {clean_text}
URL: {entry['url']}
"""
        formatted_entries.append(context.strip())
    
    return formatted_entries

def create_rag_context(logbook, query, max_entries=10):
    """Create formatted context for RAG from ELOG search."""
    # Search for relevant entries
    entries_data = search_entries_for_rag(logbook, query, max_results=max_entries)
    
    # Format for RAG context
    formatted_contexts = format_entries_for_rag(entries_data)
    
    return {
        'query': query,
        'total_entries': len(entries_data),
        'contexts': formatted_contexts,
        'source': 'SwissFEL ELOG',
        'search_timestamp': datetime.now().isoformat()
    }
```

## Key Performance Notes

- **Search Performance**: Attribute-based searches are fastest
- **Regex Performance**: `term.*term` patterns work well, avoid complex regex
- **Bulk Operations**: Process in batches to avoid timeouts
- **Error Handling**: Always use try/except for network operations
- **Result Limits**: Use reasonable limits (10-50 results) for interactive use

## Example RAG Integration

```python
# RAG system integration example
def get_elog_context_for_query(user_query, logbook):
    """Get ELOG context for a user query in RAG system."""
    
    # Map query to ELOG search terms
    search_terms = extract_technical_terms(user_query)
    
    # Perform comprehensive search
    all_results = []
    for term in search_terms:
        results = comprehensive_search(logbook, term, max_total=20)
        all_results.extend(results)
    
    # Get entry details
    entries = []
    for msg_id in set(all_results[:10]):  # Limit and deduplicate
        message, attrs, attachments = safe_read(logbook, msg_id)
        if message:
            entries.append({
                'id': msg_id,
                'title': attrs.get('Title', ''),
                'text': message,
                'date': attrs.get('Date', ''),
                'author': attrs.get('Author', ''),
                'system': attrs.get('System', '')
            })
    
    # Format for RAG
    return format_entries_for_rag(entries)
```

This reference provides all necessary ELOG read operations optimized for RAG system integration, with robust error handling and efficient search strategies.