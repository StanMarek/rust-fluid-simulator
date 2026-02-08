// Future: Particle data export to CSV, VTK, etc.

/// Export format for particle data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Csv,
    Vtk,
}

/// Trait for exporting particle data to external formats.
pub trait ParticleExporter {
    fn export(&self, path: &std::path::Path, format: ExportFormat) -> Result<(), ExportError>;
}

#[derive(Debug)]
pub enum ExportError {
    Io(std::io::Error),
    UnsupportedFormat(ExportFormat),
}

impl From<std::io::Error> for ExportError {
    fn from(e: std::io::Error) -> Self {
        ExportError::Io(e)
    }
}
