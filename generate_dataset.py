import csv

num_contents = 20
num_styles = 400

content_files = [f"content_{i:03d}.jpg" for i in range(1, num_contents+1)]
style_files = [f"style_{i:03d}.jpg" for i in range(1, num_styles+1)]

with open("content_style_pairs.txt", "w", newline="") as f:
    writer = csv.writer(f, delimiter=' ')
    
    for content in content_files:
        content_idx = int(content.split("_")[1].split(".")[0])
        
        # 2개씩 순차적으로 배치
        for i in range(0, len(style_files)-1, 2):
            style1 = style_files[i]
            style2 = style_files[i+1]
            writer.writerow([content, style1, style2])
