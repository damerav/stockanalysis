#!/bin/bash
# Daily PostgreSQL backup for stockanalysis database
# Keeps last 7 days of backups, auto-removes older ones

BACKUP_DIR="$HOME/stockanalysis/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/stockanalysis_${TIMESTAMP}.sql.gz"
KEEP_DAYS=7

mkdir -p "$BACKUP_DIR"

# Dump and compress
docker exec postgres pg_dump -U stockapp -d stockanalysis 2>/dev/null | gzip > "$BACKUP_FILE"

if [ -s "$BACKUP_FILE" ]; then
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "[$(date)] Backup OK: $BACKUP_FILE ($SIZE)"
else
    echo "[$(date)] Backup FAILED: $BACKUP_FILE is empty"
    rm -f "$BACKUP_FILE"
    exit 1
fi

# Prune backups older than KEEP_DAYS
find "$BACKUP_DIR" -name "stockanalysis_*.sql.gz" -mtime +$KEEP_DAYS -delete
REMAINING=$(ls -1 "$BACKUP_DIR"/stockanalysis_*.sql.gz 2>/dev/null | wc -l)
echo "[$(date)] Backups retained: $REMAINING"
