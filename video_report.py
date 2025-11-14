"""
Generador de reportes en PDF para videos procesados.
Acumula las detecciones por minuto y clase, y produce un resumen general.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
    from reportlab.lib.utils import ImageReader  # type: ignore
except ImportError as exc:  # pragma: no cover - fallback para entornos sin dependencia
    raise ImportError(
        "La dependencia 'reportlab' es requerida para generar el reporte PDF. "
        "Instalala ejecutando 'pip install reportlab'."
    ) from exc


@dataclass
class MinuteClassStats:
    """Estadísticas acumuladas para una clase en un minuto concreto."""

    count: int = 0
    ids: set[int] = field(default_factory=set)
    detections_without_id: int = 0

    def register_detection(self, track_id: Optional[int]) -> None:
        self.count += 1
        if track_id is None:
            self.detections_without_id += 1
        else:
            self.ids.add(track_id)


class VideoReportBuilder:
    """
    Acumula información de detecciones para generar un reporte PDF.

    - Agrupa detecciones por minuto y clase.
    - Calcula estadísticas globales por clase.
    - Genera un PDF con resumen y detalle minuto a minuto.
    """

    def __init__(self, video_filename: str, fps: float, total_frames: int) -> None:
        self.video_filename = video_filename
        self.video_stem = Path(video_filename).stem
        self.fps = fps if fps and fps > 0 else None
        self.total_frames = total_frames
        self.processed_frames = 0
        self.base_dir = Path(__file__).parent
        self.logo_path = self.base_dir / "uploads" / "usal-logo.jpg"

        # Estructuras de acumulación
        self.minute_stats: Dict[int, Dict[str, MinuteClassStats]] = defaultdict(
            lambda: defaultdict(MinuteClassStats)
        )
        self.total_detections = 0
        self.total_detections_by_class: Dict[str, int] = defaultdict(int)
        self.unique_ids_by_class: Dict[str, set[int]] = defaultdict(set)

    def add_frame_detections(
        self, frame_number: int, detections: Iterable[dict]
    ) -> None:
        """Agregar detecciones de un frame al acumulador."""
        minute_index = self._calculate_minute_index(frame_number)

        for det in detections:
            class_name = det.get("class_name", "desconocido")
            track_id = det.get("track_id")

            class_stats = self.minute_stats[minute_index][class_name]
            class_stats.register_detection(track_id if isinstance(track_id, int) else None)

            self.total_detections += 1
            self.total_detections_by_class[class_name] += 1

            if isinstance(track_id, int):
                self.unique_ids_by_class[class_name].add(track_id)

    def set_processed_frames(self, processed_frames: int) -> None:
        self.processed_frames = processed_frames

    def generate_pdf(self, output_dir: str | Path) -> Path:
        """Generar el reporte en PDF y devolver la ruta resultante."""
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        report_filename = f"{self.video_stem}_report.pdf"
        report_path = output_dir_path / report_filename

        pdf = canvas.Canvas(str(report_path), pagesize=A4)
        width, height = A4
        margin = 36  # 0.5 inch

        # Portada
        self._draw_cover_page(pdf, width, height, margin)
        pdf.showPage()

        y_position = height - margin
        line_height = 14

        def write_line(text: str, size: int = 11, bold: bool = False) -> None:
            nonlocal y_position
            if y_position <= margin:
                pdf.showPage()
                y_position = height - margin
            font_name = "Helvetica-Bold" if bold else "Helvetica"
            pdf.setFont(font_name, size)
            pdf.drawString(margin, y_position, text)
            y_position -= line_height

        # Encabezado
        write_line("Reporte de video procesado", size=16, bold=True)
        write_line(f"Archivo: {self.video_filename}", size=12)
        write_line(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", size=10)
        write_line("")

        # Resumen general
        write_line("Resumen general", size=13, bold=True)
        if self.fps:
            write_line(f"FPS estimado: {self.fps:.2f}")
        else:
            write_line("FPS estimado: no disponible")

        write_line(
            f"Frames procesados: {self.processed_frames} / {self.total_frames or 'N/D'}"
        )
        duration_text = self._format_duration()
        write_line(f"Duracion estimada: {duration_text}")
        write_line(f"Total de detecciones: {self.total_detections}")
        write_line("")

        if self.total_detections_by_class:
            write_line("Totales por clase:", bold=True)
            for class_name in sorted(self.total_detections_by_class.keys()):
                total = self.total_detections_by_class[class_name]
                unique_ids = len(self.unique_ids_by_class.get(class_name, []))
                write_line(
                    f"  - {class_name}: {total} detecciones, {unique_ids} IDs unicos"
                )
            write_line("")

        # Detalle por minuto
        write_line("Detalle por minuto", size=13, bold=True)
        if not self.minute_stats:
            write_line("No se registraron detecciones.")
        else:
            for minute_index in sorted(self.minute_stats.keys()):
                minute_label = self._format_minute_range(minute_index)
                write_line(f"Minuto {minute_label}", bold=True)

                class_mapping = self.minute_stats[minute_index]
                for class_name in sorted(class_mapping.keys()):
                    stats = class_mapping[class_name]
                    ids_sorted = sorted(stats.ids)
                    ids_text = ", ".join(str(track_id) for track_id in ids_sorted) if ids_sorted else "sin IDs"
                    extra = ""
                    if stats.detections_without_id:
                        extra = f" ({stats.detections_without_id} detecciones sin ID)"
                    write_line(
                        f"  - {class_name}: {stats.count} detecciones; IDs: {ids_text}{extra}",
                        size=10,
                    )
                write_line("")

        pdf.save()
        return report_path

    # --------------------------------------------------------------------- #
    # Métodos auxiliares

    def _calculate_minute_index(self, frame_number: int) -> int:
        if not self.fps:
            return 0
        seconds = max(frame_number - 1, 0) / self.fps
        return int(seconds // 60)

    def _format_duration(self) -> str:
        if not self.fps or not self.total_frames:
            return "no disponible"
        total_seconds = self.total_frames / self.fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _format_minute_range(minute_index: int) -> str:
        start_seconds = minute_index * 60
        end_seconds = start_seconds + 59
        start_min = start_seconds // 60
        start_sec = start_seconds % 60
        end_min = end_seconds // 60
        end_sec = end_seconds % 60
        return f"{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}"

    def _draw_cover_page(
        self, pdf: canvas.Canvas, width: float, height: float, margin: float
    ) -> None:
        """Dibujar portada con logo y datos principales."""
        title = "Reporte de Procesamiento de Video"
        subtitle = "Facultad de Ciencias Veterinarias - Universidad del Salvador"

        pdf.setFont("Helvetica-Bold", 24)
        pdf.drawCentredString(width / 2, height - margin - 20, title)

        pdf.setFont("Helvetica", 14)
        pdf.drawCentredString(
            width / 2,
            height - margin - 50,
            f"Archivo: {self.video_filename}",
        )

        pdf.setFont("Helvetica", 12)
        pdf.drawCentredString(
            width / 2,
            height - margin - 70,
            f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        if self.logo_path.exists():
            try:
                logo_reader = ImageReader(str(self.logo_path))
                logo_width, logo_height = logo_reader.getSize()
                max_logo_width = width * 0.5
                draw_width = min(max_logo_width, float(logo_width))
                aspect_ratio = float(logo_height) / float(logo_width) if logo_width else 1.0
                draw_height = draw_width * aspect_ratio
                x_position = (width - draw_width) / 2
                y_position = (height / 2) - (draw_height / 2)
                pdf.drawImage(
                    logo_reader,
                    x_position,
                    y_position,
                    width=draw_width,
                    height=draw_height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception as logo_error:  # pragma: no cover - logging manual
                print(f"⚠️  No se pudo incluir el logo en el reporte: {logo_error}")
        else:
            print(f"⚠️  Logo no encontrado en {self.logo_path}, portada sin imagen")

        pdf.setFont("Helvetica", 12)
        pdf.drawCentredString(width / 2, margin + 60, subtitle)


