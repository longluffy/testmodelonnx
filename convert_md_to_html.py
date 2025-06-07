#!/usr/bin/env python3
"""
Markdown to HTML Converter
Converts the detection_analysis_report.md to HTML format with proper styling
"""

import re
from pathlib import Path
from datetime import datetime

def convert_md_to_html(md_file_path: str, html_file_path: str):
    """
    Convert markdown file to HTML with styling
    
    Args:
        md_file_path: Path to the markdown file
        html_file_path: Path to save the HTML file
    """
    
    # Read the markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Start building HTML
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #2980b9;
        }
        td {
            padding: 10px 8px;
            border: 1px solid #ddd;
            vertical-align: middle;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e8f4f8;
        }
        .info-section {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .info-section strong {
            color: #2c3e50;
        }
        img {
            max-width: 150px;
            max-height: 150px;
            border-radius: 4px;
            border: 1px solid #ddd;
            object-fit: cover;
        }
        .notes {
            background-color: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin-top: 30px;
        }
        .notes h2 {
            margin-top: 0;
            color: #28a745;
            border: none;
            padding: 0;
        }
        .notes ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-style: italic;
        }
        .metric-value {
            font-weight: bold;
            color: #27ae60;
        }
        .no-detection {
            color: #e74c3c;
            font-style: italic;
        }
        .detection {
            color: #27ae60;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
"""
    
    # Process the markdown content line by line
    lines = md_content.split('\n')
    in_table = False
    table_headers = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines and HTML comments
        if not line or line.startswith('<!--'):
            continue
            
        # Handle headers
        if line.startswith('# '):
            html_content += f'        <h1>{line[2:]}</h1>\n'
        elif line.startswith('## '):
            html_content += f'        <h2>{line[3:]}</h2>\n'
        elif line.startswith('### '):
            html_content += f'        <h3>{line[4:]}</h3>\n'
        
        # Handle info section
        elif line.startswith('**') and line.endswith('**'):
            # Extract the content between **
            content = line[2:-2]
            if ':' in content:
                key, value = content.split(':', 1)
                html_content += f'        <div class="info-section"><strong>{key.strip()}:</strong> {value.strip()}</div>\n'
            else:
                html_content += f'        <div class="info-section"><strong>{content}</strong></div>\n'
        
        # Handle table headers
        elif '|' in line and not in_table:
            if line.startswith('|') and line.endswith('|'):
                in_table = True
                headers = [cell.strip() for cell in line.split('|')[1:-1]]
                table_headers = headers
                html_content += '        <table>\n'
                html_content += '            <thead>\n'
                html_content += '                <tr>\n'
                for header in headers:
                    html_content += f'                    <th>{header}</th>\n'
                html_content += '                </tr>\n'
                html_content += '            </thead>\n'
                html_content += '            <tbody>\n'
        
        # Handle table separator line
        elif line.startswith('|') and '---' in line and in_table:
            continue  # Skip separator line
        
        # Handle table rows
        elif line.startswith('|') and line.endswith('|') and in_table:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            html_content += '                <tr>\n'
            for j, cell in enumerate(cells):
                # Handle images
                if cell.startswith('![') and '](' in cell and ')' in cell:
                    # Extract image markdown: ![alt](src)
                    alt_text = cell[2:cell.index('](')]
                    src = cell[cell.index('](')+2:cell.rindex(')')]
                    html_content += f'                    <td><img src="{src}" alt="{alt_text}" /></td>\n'
                # Handle special formatting
                elif cell == 'No detections':
                    html_content += f'                    <td class="no-detection">{cell}</td>\n'
                elif cell in ['CA', 'PN'] or (cell.isdigit() and j < len(table_headers) and 'Class' in table_headers[j]):
                    html_content += f'                    <td class="detection">{cell}</td>\n'
                elif j < len(table_headers) and 'Value' in table_headers[j]:
                    html_content += f'                    <td class="metric-value">{cell}</td>\n'
                else:
                    html_content += f'                    <td>{cell}</td>\n'
            html_content += '                </tr>\n'
        
        # End table if we hit a non-table line
        elif in_table and not line.startswith('|'):
            html_content += '            </tbody>\n'
            html_content += '        </table>\n'
            in_table = False
            
            # Process the current line if it's not empty
            if line:
                if line.startswith('## '):
                    html_content += f'        <h2>{line[3:]}</h2>\n'
                elif line.startswith('### '):
                    html_content += f'        <h3>{line[4:]}</h3>\n'
                elif line == '---':
                    html_content += '        <hr>\n'
                else:
                    html_content += f'        <p>{line}</p>\n'
        
        # Handle other content
        elif not in_table:
            if line == '---':
                html_content += '        <hr>\n'
            elif line.startswith('- '):
                # Handle bullet points
                if i == 0 or not lines[i-1].strip().startswith('- '):
                    html_content += '        <ul>\n'
                html_content += f'            <li>{line[2:]}</li>\n'
                if i == len(lines)-1 or not lines[i+1].strip().startswith('- '):
                    html_content += '        </ul>\n'
            elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
                html_content += f'        <div class="footer">{line[1:-1]}</div>\n'
            elif line and not line.startswith('|'):
                html_content += f'        <p>{line}</p>\n'
    
    # Close any remaining table
    if in_table:
        html_content += '            </tbody>\n'
        html_content += '        </table>\n'
    
    # Close HTML
    html_content += """    </div>
</body>
</html>"""
    
    # Write HTML file
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Successfully converted {md_file_path} to {html_file_path}")


def main():
    md_file = "detection_analysis_report.md"
    html_file = "detection_analysis_report.html"
    
    print("=" * 60)
    print("Markdown to HTML Converter")
    print("=" * 60)
    print(f"Input file: {md_file}")
    print(f"Output file: {html_file}")
    print("=" * 60)
    
    try:
        convert_md_to_html(md_file, html_file)
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📄 HTML report saved as: {html_file}")
        print(f"🌐 You can open it in any web browser to view the report.")
        
    except FileNotFoundError:
        print(f"❌ Error: {md_file} not found!")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
