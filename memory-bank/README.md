# Memory Bank

This directory stores persistent memory data for the DeepWiki system.

## Structure

- `config.json` - Configuration settings for memory management
- `data/` - Directory containing memory entries
- `index.json` - Index of all memory entries and their metadata

## Data Format

Memory entries are stored in JSON format with the following structure:

```json
{
  "id": "unique-identifier",
  "timestamp": "ISO-8601 timestamp",
  "type": "memory-type",
  "content": "memory content",
  "metadata": {
    "tags": [],
    "source": "origin of memory",
    "context": "additional context"
  }
}
```

## Usage

The memory bank is used to store and retrieve important information that needs to persist across sessions. This can include:

- User preferences
- Historical context
- Learning outcomes
- Important decisions and their rationale
- Frequently accessed information

Memory entries can be tagged and categorized for efficient retrieval and context management.