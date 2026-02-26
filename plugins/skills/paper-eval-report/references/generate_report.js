const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    AlignmentType, WidthType, ShadingType, PageNumber, PageBreak
} = require('docx');
const fs = require('fs');

// 1. 평가 JSON 데이터 읽기
const jsonFile = process.argv[2] || "eval_data.json";
if (!fs.existsSync(jsonFile)) {
    console.error(`Error: ${jsonFile} not found.`);
    process.exit(1);
}
const data = JSON.parse(fs.readFileSync(jsonFile, 'utf8'));

// 2. 판정 셀 서식 헬퍼 함수
const cellBorders = {
    top: { style: "single", size: 1, color: "BFBFBF" },
    bottom: { style: "single", size: 1, color: "BFBFBF" },
    left: { style: "single", size: 1, color: "BFBFBF" },
    right: { style: "single", size: 1, color: "BFBFBF" }
};

function verdictCell(verdict, width) {
    const config = {
        "Yes": { fill: "C6EFCE", text: "✅ Yes" },
        "Partial": { fill: "FFEB9C", text: "⚠️ Partial" },
        "No": { fill: "FFC7CE", text: "❌ No" },
        "N/A": { fill: "EDEDED", text: "➖ N/A" },
    };
    // Partial Yes 등에 유연하게 대응
    const vKey = Object.keys(config).find(k => verdict.includes(k)) || "N/A";
    const c = config[vKey];

    return new TableCell({
        width: { size: width, type: WidthType.DXA },
        shading: { fill: c.fill, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        borders: cellBorders,
        children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: c.text, font: "Arial", size: 18 })]
        })]
    });
}

function textCell(text, width, options = {}) {
    return new TableCell({
        width: { size: width, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        borders: cellBorders,
        children: [new Paragraph({
            alignment: options.align || AlignmentType.LEFT,
            children: [new TextRun({ text: text || "", font: "Arial", size: 18, bold: options.bold || false })]
        })]
    });
}

function headerRow(labels, widths) {
    return new TableRow({
        tableHeader: true,
        children: labels.map((label, i) =>
            new TableCell({
                width: { size: widths[i], type: WidthType.DXA },
                shading: { fill: "1F3864", type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    children: [new TextRun({ text: label, font: "Arial", size: 18, bold: true, color: "FFFFFF" })]
                })]
            })
        )
    });
}

// 3. 체크리스트 표 생성 함수
function createChecklistTable(items, hasCritical = false) {
    const rows = [];
    let header, widths;

    if (hasCritical) {
        header = ["번호", "항목", "Critical", "판정", "근거 / 개선 제안"];
        widths = [600, 3000, 900, 1000, 3006];
    } else {
        header = ["번호", "항목", "판정", "근거 / 개선 제안"];
        widths = [600, 3000, 1000, 3906];
    }

    rows.push(headerRow(header, widths));

    for (const item of items) {
        const detailText = `[여부] ${item.findings}\n[제안] ${item.suggestions || "-"}`;

        if (hasCritical) {
            rows.push(new TableRow({
                children: [
                    textCell(item.id, widths[0], { align: AlignmentType.CENTER }),
                    textCell(item.item, widths[1]),
                    textCell(item.critical ? "★" : "", widths[2], { align: AlignmentType.CENTER, bold: true }),
                    verdictCell(item.verdict, widths[3]),
                    textCell(detailText, widths[4])
                ]
            }));
        } else {
            rows.push(new TableRow({
                children: [
                    textCell(item.id, widths[0], { align: AlignmentType.CENTER }),
                    textCell(item.item, widths[1]),
                    verdictCell(item.verdict, widths[2]),
                    textCell(detailText, widths[3])
                ]
            }));
        }
    }

    return new Table({ width: { size: 8506, type: WidthType.DXA }, rows });
}

// 4. GRADE 표 생성 함수 (SR인 경우)
function createGradeTable(outcomes) {
    if (!outcomes || outcomes.length === 0) return [new Paragraph("GRADE 평가 결과 없음.")];

    const header = ["결과", "연구수", "비뚤림", "비일관", "비직접", "비정밀", "출판편향", "등급", "효과추정치"];
    const widths = [1500, 700, 700, 700, 700, 700, 700, 900, 906];
    const rows = [headerRow(header, widths)];

    for (const o of outcomes) {
        rows.push(new TableRow({
            children: [
                textCell(o.outcome, widths[0]),
                textCell(o.studies, widths[1], { align: AlignmentType.CENTER }),
                textCell(o.risk_of_bias, widths[2], { align: AlignmentType.CENTER }),
                textCell(o.inconsistency, widths[3], { align: AlignmentType.CENTER }),
                textCell(o.indirectness, widths[4], { align: AlignmentType.CENTER }),
                textCell(o.imprecision, widths[5], { align: AlignmentType.CENTER }),
                textCell(o.publication_bias, widths[6], { align: AlignmentType.CENTER }),
                textCell(o.certainty, widths[7], { align: AlignmentType.CENTER, bold: true }),
                textCell(o.effect_estimate, widths[8], { align: AlignmentType.CENTER })
            ]
        }));
    }

    return [new Table({ width: { size: 8506, type: WidthType.DXA }, rows })];
}

// 문서 섹션 구성
const children = [
    new Paragraph({ children: [new PageBreak()] }),
    new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 2880 },
        children: [new TextRun({ text: "논문 평가 보고서", font: "Arial", size: 48, bold: true, color: "1F3864" })]
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 240 },
        children: [new TextRun({ text: data.paper_title, font: "Times New Roman", size: 24, italics: true })]
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 480 },
        children: [new TextRun({ text: `평가일: ${new Date().toISOString().split('T')[0]}`, font: "Arial", size: 20, color: "595959" })]
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: `연구 설계: ${data.study_design}`, font: "Arial", size: 20, color: "595959" })]
    }),
    new Paragraph({ children: [new PageBreak()] }),

    new Paragraph({ text: "1. 적용된 평가 지침", style: "Heading2" }),
    new Paragraph({ text: `보고 지침: ${data.reporting_guideline_name || "N/A"}` }),
    new Paragraph({ text: `질/비뚤림 도구: ${data.quality_tool_name || "N/A"}` }),

    new Paragraph({ text: "2. 보고 지침 준수 평가", style: "Heading2" })
];

if (data.reporting_eval && data.reporting_eval.length > 0) {
    children.push(createChecklistTable(data.reporting_eval, false));
}

children.push(new Paragraph({ text: "3. 방법론적 질 / 비뚤림 위험 평가", style: "Heading2", spacing: { before: 400 } }));
if (data.quality_eval && data.quality_eval.length > 0) {
    const hasCritical = data.quality_eval.some(i => i.critical);
    children.push(createChecklistTable(data.quality_eval, hasCritical));
}

if (data.study_design.startsWith("SR")) {
    children.push(new Paragraph({ text: "4. GRADE 근거 확실성 (Evidence Profile)", style: "Heading2", spacing: { before: 400 } }));
    children.push(...createGradeTable(data.grade_eval));
}

children.push(new Paragraph({ text: "5. 종합 요약 및 권고사항", style: "Heading2", spacing: { before: 400 } }));
children.push(new Paragraph({ text: data.overall_summary || "요약 내용 없음" }));

// 5. 문서 객체 생성
const doc = new Document({
    styles: {
        default: { document: { run: { font: "Times New Roman", size: 24 } } },
        paragraphStyles: [
            {
                id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
                run: { font: "Arial", size: 24, bold: true, color: "2E4057" },
                paragraph: { spacing: { before: 240, after: 80 } }
            }
        ]
    },
    sections: [{
        properties: {
            page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 } }
        },
        children: children
    }]
});

// 6. 파일 저장
const safeTitle = data.paper_title.replace(/[\\/:*?"<>|]/g, "").substring(0, 30).trim();
const today = new Date().toISOString().split('T')[0].replace(/-/g, "");
const outputPath = `${safeTitle || "논문"}_평가보고서_${today}.docx`;

Packer.toBuffer(doc).then((buffer) => {
    fs.writeFileSync(outputPath, buffer);
    console.log(`Report successfully generated: ${outputPath}`);
}).catch(err => {
    console.error("Error generating document:", err);
});
