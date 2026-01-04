# DGUS≤6 非标准 ICO 提取工具

本项目用于提取 **DGUS 版本小于等于 6** 时生成的 **非标准 ICO** 文件中的多张图片内容。

这些 ICO 并非 Windows 标准 ICO，而更像一种“多图片容器”：头部是条目表，后面是 16bpp 的原始像素流（本项目默认按 RGB565 解码）。

## 结论（本项目默认采用的正确方式）

这些文件的条目并不是“顺序拼接”，而是 **banked（分段）寻址**：

- 像素数据区起始偏移通常是 `0x40000`（脚本会自动推断）。
- 每个条目占 8 字节：`w_h(u16) | fmt16(u16) | off_lo(u16) | off_hi(u16)`
  - `width = w_h & 0xFF`
  - `height = (w_h >> 8) & 0xFF`
- `fmt16` 的高字节决定 bank：`0x200 -> bank0`，`0x300 -> bank1`，`0x400 -> bank2` ……
  - `bank_index = (fmt16 >> 8) - 2`
  - 每个 bank 的跨度是 `0x10000` 个 word（即 `0x20000` 字节）
- `off_lo` 需要先做 16 位字节交换，再以 word 为单位参与寻址：
  - `start_words = bank_index * 0x10000 + byteswap16(off_lo)`
  - `offset_bytes = data_offset + start_words * 2`

用这套规则读取 `width*height` 个 16 位像素并按 **RGB565 + swap-bytes** 解码，就能消除“横向雪花条纹/错位切分”的问题，并得到与原图一致的结果。

## 色阶说明（重要）

本项目输出的是 RGB565（或你指定的其它 16bpp 格式）。在 RGB565 中：

- **G 通道是 6 bit**
- **R/B 通道是 5 bit**

因此 R/B 的量化更粗（可理解为“压缩率比 G 大”），更容易出现明显色阶/条带。**这种量化是不可逆的**：如果源数据本身就是 RGB565（或更低精度）存储，本程序无法恢复到更高精度的平滑渐变。

## 默认输出规则

- 默认输出到：`extracted/YYYYMMDD_HHMMSS/<ICO同名文件夹>/`
- 图片文件名按索引：`0.bmp / 1.bmp / 2.bmp ...`（不补零）

## 使用方法

### 1) 提取单个 ICO

```bash
python3 extract_custom_ico.py DWIN_SET/某个文件.ICO
```

### 2) 批量提取 DWIN_SET 下所有 ICO

```bash
python3 extract_custom_ico.py DWIN_SET/*.ICO
```

### 3) 常用可选项

- 指定输出目录：
  - `--out-dir extracted`
- 不创建时间戳目录（不推荐）：
  - `--no-timestamp-dir`
- 输出目录名带上格式后缀（默认不带）：
  - `--decorated-dir`
- 强制数据区起始（一般不需要）：
  - `--data-offset 0x40000`

## 预览提示（BMP 转 PNG）

部分环境/工具不支持直接预览 BMP。macOS 可用 `sips` 转成 PNG：

```bash
sips -s format png extracted/某次时间戳/某个ICO名/7.bmp --out 7.png
```
