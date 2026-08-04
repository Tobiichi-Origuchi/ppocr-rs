//! Versioned, bounded IPC messages shared by the GTK frontend and OCR worker.

mod codec;
mod types;

pub use codec::{
    ATTACHMENT_FLAG, CodecError, Frame, HEADER_LEN, Limits, MAGIC, PROTOCOL_MAJOR, read_frame,
    read_frame_with_limits, write_frame, write_frame_with_limits,
};
pub use types::*;
