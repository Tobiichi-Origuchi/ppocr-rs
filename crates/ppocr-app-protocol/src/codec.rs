use std::io::{self, Read, Write};

use thiserror::Error;

use crate::{Message, ProtocolLimits};

pub const MAGIC: [u8; 4] = *b"POCR";
pub const PROTOCOL_MAJOR: u16 = 1;
pub const ATTACHMENT_FLAG: u16 = 1;
pub const HEADER_LEN: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_json_bytes: u32,
    pub max_attachment_bytes: u64,
}

impl Limits {
    pub const DEFAULT: Self = Self {
        max_json_bytes: 16 * 1024 * 1024,
        max_attachment_bytes: 256 * 1024 * 1024,
    };

    pub const fn protocol_limits(self) -> ProtocolLimits {
        ProtocolLimits {
            max_json_bytes: self.max_json_bytes,
            max_attachment_bytes: self.max_attachment_bytes,
            max_image_dimension: 32_768,
            max_image_pixels: 64 * 1024 * 1024,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub message: Message,
    pub attachment: Vec<u8>,
}

impl Frame {
    pub fn new(message: Message) -> Self {
        Self {
            message,
            attachment: Vec::new(),
        }
    }

    pub fn with_attachment(message: Message, attachment: Vec<u8>) -> Self {
        Self {
            message,
            attachment,
        }
    }
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("IPC I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("invalid IPC magic")]
    InvalidMagic,
    #[error("unsupported IPC protocol major {actual}; expected {expected}")]
    UnsupportedMajor { expected: u16, actual: u16 },
    #[error("unknown IPC frame flags 0x{0:04x}")]
    UnknownFlags(u16),
    #[error("IPC attachment flag does not match attachment length")]
    InconsistentAttachment,
    #[error("IPC JSON length {actual} exceeds limit {limit}")]
    JsonTooLarge { limit: u32, actual: u32 },
    #[error("IPC attachment length {actual} exceeds limit {limit}")]
    AttachmentTooLarge { limit: u64, actual: u64 },
    #[error("IPC JSON cannot be represented in a u32 length")]
    JsonLengthOverflow,
    #[error("IPC attachment cannot be represented in a u64 length")]
    AttachmentLengthOverflow,
    #[error("failed to encode IPC JSON: {0}")]
    EncodeJson(#[source] serde_json::Error),
    #[error("failed to decode IPC JSON: {0}")]
    DecodeJson(#[source] serde_json::Error),
}

pub fn write_frame(writer: &mut impl Write, frame: &Frame) -> Result<(), CodecError> {
    write_frame_with_limits(writer, frame, Limits::DEFAULT)
}

pub fn write_frame_with_limits(
    writer: &mut impl Write,
    frame: &Frame,
    limits: Limits,
) -> Result<(), CodecError> {
    let json = serde_json::to_vec(&frame.message).map_err(CodecError::EncodeJson)?;
    let json_len = u32::try_from(json.len()).map_err(|_| CodecError::JsonLengthOverflow)?;
    let attachment_len =
        u64::try_from(frame.attachment.len()).map_err(|_| CodecError::AttachmentLengthOverflow)?;
    check_lengths(json_len, attachment_len, limits)?;

    let flags = if attachment_len == 0 {
        0
    } else {
        ATTACHMENT_FLAG
    };
    let mut header = [0_u8; HEADER_LEN];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&PROTOCOL_MAJOR.to_le_bytes());
    header[6..8].copy_from_slice(&flags.to_le_bytes());
    header[8..12].copy_from_slice(&json_len.to_le_bytes());
    header[12..20].copy_from_slice(&attachment_len.to_le_bytes());
    writer.write_all(&header)?;
    writer.write_all(&json)?;
    writer.write_all(&frame.attachment)?;
    writer.flush()?;
    Ok(())
}

pub fn read_frame(reader: &mut impl Read) -> Result<Option<Frame>, CodecError> {
    read_frame_with_limits(reader, Limits::DEFAULT)
}

pub fn read_frame_with_limits(
    reader: &mut impl Read,
    limits: Limits,
) -> Result<Option<Frame>, CodecError> {
    let mut header = [0_u8; HEADER_LEN];
    let first = loop {
        match reader.read(&mut header[..1]) {
            Ok(0) => return Ok(None),
            Ok(1) => break header[0],
            Ok(_) => unreachable!("one-byte read returned more than one byte"),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    };
    header[0] = first;
    reader.read_exact(&mut header[1..])?;
    if header[0..4] != MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    let major = u16::from_le_bytes([header[4], header[5]]);
    if major != PROTOCOL_MAJOR {
        return Err(CodecError::UnsupportedMajor {
            expected: PROTOCOL_MAJOR,
            actual: major,
        });
    }
    let flags = u16::from_le_bytes([header[6], header[7]]);
    if flags & !ATTACHMENT_FLAG != 0 {
        return Err(CodecError::UnknownFlags(flags));
    }
    let json_len = u32::from_le_bytes(header[8..12].try_into().expect("fixed header slice"));
    let attachment_len = u64::from_le_bytes(header[12..20].try_into().expect("fixed header slice"));
    let has_attachment = flags & ATTACHMENT_FLAG != 0;
    if has_attachment != (attachment_len > 0) {
        return Err(CodecError::InconsistentAttachment);
    }
    check_lengths(json_len, attachment_len, limits)?;

    let mut json = vec![0_u8; json_len as usize];
    reader.read_exact(&mut json)?;
    let message = serde_json::from_slice(&json).map_err(CodecError::DecodeJson)?;
    let attachment_usize =
        usize::try_from(attachment_len).map_err(|_| CodecError::AttachmentTooLarge {
            limit: limits.max_attachment_bytes,
            actual: attachment_len,
        })?;
    let mut attachment = vec![0_u8; attachment_usize];
    reader.read_exact(&mut attachment)?;
    Ok(Some(Frame {
        message,
        attachment,
    }))
}

fn check_lengths(json_len: u32, attachment_len: u64, limits: Limits) -> Result<(), CodecError> {
    if json_len > limits.max_json_bytes {
        return Err(CodecError::JsonTooLarge {
            limit: limits.max_json_bytes,
            actual: json_len,
        });
    }
    if attachment_len > limits.max_attachment_bytes {
        return Err(CodecError::AttachmentTooLarge {
            limit: limits.max_attachment_bytes,
            actual: attachment_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read};

    use super::*;
    use crate::{PROTOCOL_MINOR, Ping};

    struct Fragmented<R> {
        inner: R,
    }

    impl<R: Read> Read for Fragmented<R> {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let length = output.len().min(1);
            self.inner.read(&mut output[..length])
        }
    }

    fn ping_frame() -> Frame {
        Frame::with_attachment(Message::Ping(Ping { request_id: 42 }), vec![1, 2, 3, 4])
    }

    #[test]
    fn round_trip_survives_one_byte_reads() {
        let expected = ping_frame();
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &expected).unwrap();
        let mut reader = Fragmented {
            inner: Cursor::new(bytes),
        };
        assert_eq!(read_frame(&mut reader).unwrap(), Some(expected));
        assert_eq!(read_frame(&mut reader).unwrap(), None);
    }

    #[test]
    fn header_is_stable_and_little_endian() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &ping_frame()).unwrap();
        assert_eq!(&bytes[0..4], b"POCR");
        assert_eq!(&bytes[4..6], &PROTOCOL_MAJOR.to_le_bytes());
        assert_eq!(&bytes[6..8], &ATTACHMENT_FLAG.to_le_bytes());
        assert_eq!(u64::from_le_bytes(bytes[12..20].try_into().unwrap()), 4);
        assert!(String::from_utf8_lossy(&bytes[HEADER_LEN..]).contains("request_id"));
        assert_eq!(PROTOCOL_MINOR, 0);
    }

    #[test]
    fn rejects_bad_magic_unknown_flags_and_major() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Frame::new(Message::Ping(Ping { request_id: 1 })),
        )
        .unwrap();

        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            read_frame(&mut Cursor::new(bad_magic)),
            Err(CodecError::InvalidMagic)
        ));

        let mut bad_flags = bytes.clone();
        bad_flags[6..8].copy_from_slice(&2_u16.to_le_bytes());
        assert!(matches!(
            read_frame(&mut Cursor::new(bad_flags)),
            Err(CodecError::UnknownFlags(2))
        ));

        let mut bad_major = bytes;
        bad_major[4..6].copy_from_slice(&9_u16.to_le_bytes());
        assert!(matches!(
            read_frame(&mut Cursor::new(bad_major)),
            Err(CodecError::UnsupportedMajor { actual: 9, .. })
        ));
    }

    #[test]
    fn rejects_hostile_lengths_before_allocation() {
        let mut header = [0_u8; HEADER_LEN];
        header[0..4].copy_from_slice(&MAGIC);
        header[4..6].copy_from_slice(&PROTOCOL_MAJOR.to_le_bytes());
        header[8..12].copy_from_slice(&101_u32.to_le_bytes());
        let limits = Limits {
            max_json_bytes: 100,
            max_attachment_bytes: 200,
        };
        assert!(matches!(
            read_frame_with_limits(&mut Cursor::new(header), limits),
            Err(CodecError::JsonTooLarge { actual: 101, .. })
        ));

        header[8..12].copy_from_slice(&0_u32.to_le_bytes());
        header[6..8].copy_from_slice(&ATTACHMENT_FLAG.to_le_bytes());
        header[12..20].copy_from_slice(&201_u64.to_le_bytes());
        assert!(matches!(
            read_frame_with_limits(&mut Cursor::new(header), limits),
            Err(CodecError::AttachmentTooLarge { actual: 201, .. })
        ));
    }

    #[test]
    fn rejects_truncated_and_inconsistent_frames() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &ping_frame()).unwrap();
        bytes.pop();
        assert!(matches!(
            read_frame(&mut Cursor::new(bytes)),
            Err(CodecError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof
        ));

        let mut header = [0_u8; HEADER_LEN];
        header[0..4].copy_from_slice(&MAGIC);
        header[4..6].copy_from_slice(&PROTOCOL_MAJOR.to_le_bytes());
        header[12..20].copy_from_slice(&1_u64.to_le_bytes());
        assert!(matches!(
            read_frame(&mut Cursor::new(header)),
            Err(CodecError::InconsistentAttachment)
        ));
    }

    #[test]
    fn write_obeys_custom_limits() {
        let limits = Limits {
            max_json_bytes: 1,
            max_attachment_bytes: 3,
        };
        assert!(matches!(
            write_frame_with_limits(&mut Vec::new(), &ping_frame(), limits),
            Err(CodecError::JsonTooLarge { .. })
        ));
    }
}
