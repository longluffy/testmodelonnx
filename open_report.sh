#!/bin/bash
# Open the HTML detection analysis report in the default browser

HTML_FILE="detection_analysis_report.html"

if [ -f "$HTML_FILE" ]; then
    echo "🌐 Opening Detection Analysis Report in browser..."
    
    # Try different browsers/commands
    if command -v xdg-open > /dev/null; then
        xdg-open "$HTML_FILE"
    elif command -v open > /dev/null; then
        open "$HTML_FILE"
    elif command -v firefox > /dev/null; then
        firefox "$HTML_FILE" &
    elif command -v google-chrome > /dev/null; then
        google-chrome "$HTML_FILE" &
    elif command -v chromium-browser > /dev/null; then
        chromium-browser "$HTML_FILE" &
    else
        echo "❌ No suitable browser found. Please open $HTML_FILE manually."
        echo "📁 Full path: $(pwd)/$HTML_FILE"
    fi
    
    echo "✅ Report opened successfully!"
else
    echo "❌ Error: $HTML_FILE not found!"
    echo "Please run the analysis first: python3 run_analysis.py"
    echo "Then convert to HTML: python3 convert_md_to_html.py"
fi
