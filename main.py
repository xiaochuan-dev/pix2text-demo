import os
import zipfile
from pathlib import Path
from typing import List, Union
from PIL import Image

from huggingface_hub import hf_hub_download
import pypdfium2 as pdfium
from pix2text import Pix2Text

# ==================== 1. 下载 + 解压 ====================
def download_and_unzip_pdfs(
    repo_id: str = "xiaochuan-dev/pdf",
    filename: str = "pdf.zip",
    local_dir: str = "./downloaded_pdfs",
):
    """
    从 Hugging Face 下载 zip 并解压，返回所有 PDF 路径列表
    """
    os.makedirs(local_dir, exist_ok=True)

    print(f"正在从 Hugging Face 下载 {filename} ...")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",          # 重要！因为是 datasets 仓库
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"下载完成: {zip_path}")

    # 解压
    extract_dir = Path(local_dir) / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"解压完成，文件保存在: {extract_dir}")

    # 收集所有 PDF
    pdf_files = list(extract_dir.rglob("*.pdf"))
    print(f"共找到 {len(pdf_files)} 个 PDF 文件")
    return pdf_files

# ==================== 2. PDF 渲染成图片 ====================
def pdf_to_images(
    pdf_path: Union[str, Path],
    output_dir: str = "./pdf_images",
    dpi: int = 200,                 # 推荐 150~300，越高越清晰但越慢
    max_pages: int = None,          # 限制页数，None 表示全部
) -> List[Path]:
    """
    把单个 PDF 渲染成图片列表，返回图片路径
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    image_paths = []

    total_pages = len(pdf)
    pages_to_render = min(total_pages, max_pages) if max_pages else total_pages

    print(f"  正在渲染 PDF: {pdf_path.name} （共 {total_pages} 页，处理前 {pages_to_render} 页）")

    scale = dpi / 72  # pypdfium2 的 scale 是相对于 72 DPI 的倍数

    for i in range(pages_to_render):
        page = pdf[i]
        # 渲染
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        img_path = out_dir / f"page_{i+1:03d}.png"
        pil_image.save(img_path, "PNG")
        image_paths.append(img_path)

    pdf.close()
    return image_paths

# ==================== 3. 公式检测 + 切割 + 识别 ====================
def extract_formulas_from_images(
    image_list: List[Union[str, Path, Image.Image]],
    output_dir: str = "formula_crops",
    resized_shape: int = 768,
    save_crops: bool = True,
):
    """
    从图片列表中检测公式、切割公式图片，并返回 LaTeX
    """
    if save_crops:
        os.makedirs(output_dir, exist_ok=True)

    p2t = Pix2Text.from_config()   # 使用最新的 mfd-1.5 + mfr-1.5

    all_results = []

    for img_idx, img in enumerate(image_list):
        print(f"\n正在处理第 {img_idx + 1}/{len(image_list)} 张图片...")

        outs = p2t.recognize(
            img,
            resized_shape=resized_shape,
            return_text=False,
            contain_formula=True,
        )

        if isinstance(img, (str, Path)):
            original_img = Image.open(img).convert("RGB")
            img_name = Path(img).stem
        else:
            original_img = img.convert("RGB")
            img_name = f"image_{img_idx:03d}"

        formula_results = []

        for box_idx, box in enumerate(outs):
            if box.get("type") in ("isolated", "embedding", "formula"):
                position = box["position"]
                xs = [p[0] for p in position]
                ys = [p[1] for p in position]
                left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)

                # 稍微扩大边界
                margin = 6
                left = max(0, left - margin)
                top = max(0, top - margin)
                right = min(original_img.width, right + margin)
                bottom = min(original_img.height, bottom + margin)

                formula_img = original_img.crop((left, top, right, bottom))

                crop_path = None
                if save_crops:
                    crop_filename = f"{img_name}_formula_{box_idx:02d}.png"
                    crop_path = os.path.join(output_dir, crop_filename)
                    formula_img.save(crop_path)

                latex = box.get("text", "").strip()

                formula_results.append({
                    "latex": latex,
                    "type": box.get("type"),
                    "score": box.get("score"),
                    "bbox": [left, top, right, bottom],
                    "crop_path": crop_path,
                })

        all_results.append({
            "image_index": img_idx,
            "image_name": img_name,
            "formulas": formula_results,
            "formula_count": len(formula_results),
        })
        print(f"  → 检测到 {len(formula_results)} 个公式")

    return all_results

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 下载并解压
    pdf_files = download_and_unzip_pdfs(
        repo_id="xiaochuan-dev/pdf",
        filename="pdf.zip",
        local_dir="./downloaded_pdfs",
    )

    # 2. 把所有 PDF 渲染成图片
    all_image_paths = []
    for pdf in pdf_files:
        images = pdf_to_images(
            pdf_path=pdf,
            output_dir="./pdf_images",
            dpi=200,            # 可根据需要调整，推荐 150~250
            max_pages=None,     # 想先测试可以改成 5 或 10
        )
        all_image_paths.extend(images)

    print(f"\n总共生成了 {len(all_image_paths)} 张图片")

    # 3. 公式识别 + 切割
    results = extract_formulas_from_images(
        image_list=all_image_paths,
        output_dir="formula_crops",
        resized_shape=768,
        save_crops=True,
    )

    # 4. 打印结果
    total_formulas = 0
    for res in results:
        total_formulas += res["formula_count"]
        if res["formula_count"] > 0:
            print(f"\n===== {res['image_name']} =====")
            for i, f in enumerate(res["formulas"], 1):
                print(f"【公式 {i}】{f['type']}")
                print(f"LaTeX: {f['latex']}")
                print(f"保存: {f['crop_path']}")

    print(f"\n全部完成！共提取到 {total_formulas} 个公式。")