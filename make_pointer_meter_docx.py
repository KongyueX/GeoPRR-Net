import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape

parser = argparse.ArgumentParser(description="生成仪表识别整体流程说明文档。")
parser.add_argument(
    "--output",
    type=Path,
    default=Path("仪表识别整体流程说明.docx"),
)
out = parser.parse_args().output
out.parent.mkdir(parents=True, exist_ok=True)

paragraphs = [
    "仪表识别整体流程说明",
    "",
    "1. 项目总体定位",
    "这是一个基于 FastAPI 的指针式仪表读数识别服务。系统接收图片路径、base64 图片或者摄像头画面，先把输入统一转成 OpenCV 图像，再调用 PointerMeterInferModel 完成表盘检测、图像校正、指针分割、读数推理和结果返回。",
    "",
    "2. 服务层整体流程",
    "2.1 请求入口",
    "主接口是 POST /infer。请求体主要包含 modelName、inferData、dataType、cameraTimeout 和 inferConfig。当前实际生产模型名是 PointerMeterInferModel。",
    "2.2 输入数据处理",
    "DataProcessService 负责把不同来源的数据统一转成 OpenCV 图像：filepath 走 cv2.imread，base64 走 base64 解码加 cv2.imdecode，cameraId 走 CameraProcessService 获取摄像头帧。",
    "2.3 模型动态加载",
    "main.py 会读取 models/model_list.yaml，根据 modelName 动态导入模型类，并把模型实例缓存成单例，避免每次请求重复加载权重。",
    "",
    "3. 核心识别流程",
    "真正的识别主流程在 meterZeroShot.Inference 中，整体可以分成以下几个阶段。",
    "3.1 表盘检测与裁剪",
    "系统先用 YOLO 在整张图上检测表盘区域，并裁剪出表盘 crop。如果没有检测到表盘，就直接返回“未检测到表盘”。",
    "3.2 图像校正",
    "得到表盘 crop 后，系统会根据 correction_mode 决定是否做校正。支持 off、ellipse、ransacFun、ransacFunbackup、square、stretch 等模式。",
    "其中 ellipse 是当前比较重要的自动矫正方案：它会先在裁剪表盘上锁定白色表盘面，拟合椭圆，再尝试用 YOLO 检测到的中心点替换椭圆中心，最后通过仿射变换把椭圆压回接近正圆，用来减小斜拍带来的读数误差。",
    "3.3 指针分割",
    "校正后的表盘图会送入 U2NetP 分割模型，得到指针 mask。然后程序只保留最大连通域，尽量去掉噪声区域。",
    "3.4 指针 mask 主轴校验",
    "如果 inferConfig 中开启 validate_mask_line，系统会对 mask 做 cv2.fitLine 主轴拟合，并检查这条主轴延长线是否经过表盘中心附近。如果距离中心太远，就认为当前 mask 不像真实指针，返回“无法找到指针”。",
    "3.5 归一化刻度识别",
    "接下来系统会把指针 mask 和表盘图同时输入 meterFormer / meterClip 模型。这个模型不是直接回归浮点数，而是把指针位置离散成 0 到 100 共 101 个刻度类别，再通过图像特征和文本标签“0~100”的匹配关系，预测当前最接近哪一个刻度，输出 endNum。",
    "3.6 起点和终点检测",
    "有了 endNum 还不够，因为不同表盘的真实量程不同。系统还会用另一个 YOLO 模型在表盘图上检测起点和终点位置。",
    "如果起点和终点都检测到了，就直接按它们的角度关系计算量程角；如果只检测到一个点，就用 default_range_angle 推算另一个点；如果两个点都没检测到，就退回 default_start_angle 和 default_range_angle 这组默认参数。",
    "3.7 真实读数换算",
    "系统先把 endNum 转成指针在表盘上的相对角度，再结合起点角、终点角、量程角以及 scaleStart 和 scaleEnd，把归一化刻度换算成真实物理读数 result。",
    "",
    "4. 各子模型的作用",
    "4.1 表盘检测 YOLO：从整图中定位表盘区域并裁剪。",
    "4.2 指针分割 U2NetP：从表盘图中分割出指针 mask。",
    "4.3 读数识别 meterFormer / meterClip：输入指针 mask 和表盘图，输出 0~100 的归一化刻度 endNum。",
    "4.4 起点终点检测 YOLO：检测表盘上的起点、终点，ellipse 矫正时还可辅助中心点定位。",
    "",
    "5. 训练流程总结",
    "需要特别说明：这个仓库主要是推理与服务仓库，不是完整训练仓库。仓库里能看到权重加载逻辑、数据加载器结构和模型推理结构，但没有完整 train.py、optimizer、loss、scheduler 等正式训练脚本。",
    "不过从代码结构可以清楚推断出，这套方案是分模块训练、推理时串联。",
    "5.1 表盘检测模型训练：训练 YOLO 学会在整图中定位表盘，输出权重 yolo_findMeter.pt。",
    "5.2 指针分割模型训练：训练 U2NetP 从表盘图中分割指针，输出权重 pointerSeg/resultSeg/best.pt。",
    "5.3 起终点检测模型训练：训练 YOLO 学会识别表盘上的起点、终点和中心点，输出权重 yolo_pointbest.pt。",
    "5.4 读数识别模型训练：训练 meterFormer / meterClip 输入“指针 mask + 表盘图”，输出 0~100 的离散刻度标签，输出权重 vitTranforms/result/best.pt。",
    "5.5 推理串联：实际部署时不是单一模型端到端输出，而是“表盘检测 -> 图像校正 -> 指针分割 -> 归一化刻度识别 -> 起终点检测 -> 真实读数换算”的多阶段流水线。",
    "",
    "6. 输入输出说明",
    "6.1 API 输入",
    "请求体至少包含 modelName、inferData、dataType 和 inferConfig。inferData 可以是图片路径、base64 字符串或者 cameraId。",
    "6.2 API 输出",
    "成功时返回 status=true、message、result，以及可选的 result_pointer_image 和 result_mask_image；失败时返回 status=false、失败原因和空结果。",
    "6.3 模型层输入输出",
    "表盘检测模型输入整图，输出表盘 crop；指针分割模型输入表盘图，输出 mask；读数识别模型输入 mask 和表盘图，输出 endNum；起终点检测模型输入表盘图，输出参考点坐标；最后换算模块输入 endNum、参考点和量程，输出真实读数。",
    "",
    "7. inferConfig 的作用",
    "inferConfig 是整个工程的关键调参入口。常用参数包括：scaleStart / scaleEnd（量程范围）、correction_mode（图像校正方式）、confidence（YOLO 阈值）、reading_offset（刻度微调）、validate_mask_line（mask 校验）、auto_zero（自动归零）、result_pointer_image / result_mask_image（是否返回调试图）。",
    "",
    "8. 方案优点与局限",
    "优点：模块化清晰、参数可控、工程化部署方便、支持图片和摄像头输入。",
    "局限：仓库内训练链路不完整；多阶段流水线会产生误差传递；最终读数对起终点检测较依赖；当前服务默认在 CPU 上运行。",
    "",
    "9. 最终总结",
    "这套指针仪表识别系统不是单一端到端模型，而是由“YOLO 表盘检测 + U2Net 指针分割 + Transformer/CLIP 式归一化刻度识别 + YOLO 起终点检测”组合而成。系统先找到表盘，再找到指针，再预测指针所在刻度，最后结合量程参考点换算出真实读数。"
]


def para_xml(text: str) -> str:
    if text == "":
        return "<w:p/>"
    safe = escape(text)
    return f'<w:p><w:r><w:t xml:space="preserve">{safe}</w:t></w:r></w:p>'


document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:w10="urn:schemas-microsoft-com:office:word" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" xmlns:wne="http://schemas.microsoft.com/office/2006/wordml" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" mc:Ignorable="w14 wp14">
  <w:body>
    {' '.join(para_xml(p) for p in paragraphs)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
      <w:cols w:space="708"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>
  </w:body>
</w:document>
'''

content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
'''

rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
'''

core = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>仪表识别整体流程说明</dc:title>
  <dc:creator>Claude Code</dc:creator>
  <cp:lastModifiedBy>Claude Code</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-06-29T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-29T00:00:00Z</dcterms:modified>
</cp:coreProperties>
'''

app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Claude Code</Application>
</Properties>
'''

with ZipFile(out, 'w', ZIP_DEFLATED) as zf:
    zf.writestr('[Content_Types].xml', content_types)
    zf.writestr('_rels/.rels', rels)
    zf.writestr('word/document.xml', document_xml)
    zf.writestr('docProps/core.xml', core)
    zf.writestr('docProps/app.xml', app)

print(out)
