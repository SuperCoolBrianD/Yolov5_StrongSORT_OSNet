# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from retina_view/MsgRadarPoint.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import retina_view.msg

class MsgRadarPoint(genpy.Message):
  _md5sum = "a89e8d5e2a555f3edaf778b1c884cef3"
  _type = "retina_view/MsgRadarPoint"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """int32 position
int32 nTargets
Point[] points
uint32 firmware_version
int32 frame_counter
uint32[] time_stamp
uint32 track_count  
Track[] track_info

================================================================================
MSG: retina_view/Point
float32 x
float32 y
float32 z
float32 doppler
float32 power

uint8 track

================================================================================
MSG: retina_view/Track
uint32 track_id
uint8 track_state
uint8 track_count
uint16 track_range
int8 track_angle
uint8 track_type
uint8 track_activity
uint8 track_reliability
"""
  __slots__ = ['position','nTargets','points','firmware_version','frame_counter','time_stamp','track_count','track_info']
  _slot_types = ['int32','int32','retina_view/Point[]','uint32','int32','uint32[]','uint32','retina_view/Track[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       position,nTargets,points,firmware_version,frame_counter,time_stamp,track_count,track_info

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(MsgRadarPoint, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.position is None:
        self.position = 0
      if self.nTargets is None:
        self.nTargets = 0
      if self.points is None:
        self.points = []
      if self.firmware_version is None:
        self.firmware_version = 0
      if self.frame_counter is None:
        self.frame_counter = 0
      if self.time_stamp is None:
        self.time_stamp = []
      if self.track_count is None:
        self.track_count = 0
      if self.track_info is None:
        self.track_info = []
    else:
      self.position = 0
      self.nTargets = 0
      self.points = []
      self.firmware_version = 0
      self.frame_counter = 0
      self.time_stamp = []
      self.track_count = 0
      self.track_info = []

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_2i().pack(_x.position, _x.nTargets))
      length = len(self.points)
      buff.write(_struct_I.pack(length))
      for val1 in self.points:
        _x = val1
        buff.write(_get_struct_5fB().pack(_x.x, _x.y, _x.z, _x.doppler, _x.power, _x.track))
      _x = self
      buff.write(_get_struct_Ii().pack(_x.firmware_version, _x.frame_counter))
      length = len(self.time_stamp)
      buff.write(_struct_I.pack(length))
      pattern = '<%sI'%length
      buff.write(struct.Struct(pattern).pack(*self.time_stamp))
      _x = self.track_count
      buff.write(_get_struct_I().pack(_x))
      length = len(self.track_info)
      buff.write(_struct_I.pack(length))
      for val1 in self.track_info:
        _x = val1
        buff.write(_get_struct_I2BHb3B().pack(_x.track_id, _x.track_state, _x.track_count, _x.track_range, _x.track_angle, _x.track_type, _x.track_activity, _x.track_reliability))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.points is None:
        self.points = None
      if self.track_info is None:
        self.track_info = None
      end = 0
      _x = self
      start = end
      end += 8
      (_x.position, _x.nTargets,) = _get_struct_2i().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.points = []
      for i in range(0, length):
        val1 = retina_view.msg.Point()
        _x = val1
        start = end
        end += 21
        (_x.x, _x.y, _x.z, _x.doppler, _x.power, _x.track,) = _get_struct_5fB().unpack(str[start:end])
        self.points.append(val1)
      _x = self
      start = end
      end += 8
      (_x.firmware_version, _x.frame_counter,) = _get_struct_Ii().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sI'%length
      start = end
      s = struct.Struct(pattern)
      end += s.size
      self.time_stamp = s.unpack(str[start:end])
      start = end
      end += 4
      (self.track_count,) = _get_struct_I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.track_info = []
      for i in range(0, length):
        val1 = retina_view.msg.Track()
        _x = val1
        start = end
        end += 12
        (_x.track_id, _x.track_state, _x.track_count, _x.track_range, _x.track_angle, _x.track_type, _x.track_activity, _x.track_reliability,) = _get_struct_I2BHb3B().unpack(str[start:end])
        self.track_info.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_2i().pack(_x.position, _x.nTargets))
      length = len(self.points)
      buff.write(_struct_I.pack(length))
      for val1 in self.points:
        _x = val1
        buff.write(_get_struct_5fB().pack(_x.x, _x.y, _x.z, _x.doppler, _x.power, _x.track))
      _x = self
      buff.write(_get_struct_Ii().pack(_x.firmware_version, _x.frame_counter))
      length = len(self.time_stamp)
      buff.write(_struct_I.pack(length))
      pattern = '<%sI'%length
      buff.write(self.time_stamp.tostring())
      _x = self.track_count
      buff.write(_get_struct_I().pack(_x))
      length = len(self.track_info)
      buff.write(_struct_I.pack(length))
      for val1 in self.track_info:
        _x = val1
        buff.write(_get_struct_I2BHb3B().pack(_x.track_id, _x.track_state, _x.track_count, _x.track_range, _x.track_angle, _x.track_type, _x.track_activity, _x.track_reliability))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.points is None:
        self.points = None
      if self.track_info is None:
        self.track_info = None
      end = 0
      _x = self
      start = end
      end += 8
      (_x.position, _x.nTargets,) = _get_struct_2i().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.points = []
      for i in range(0, length):
        val1 = retina_view.msg.Point()
        _x = val1
        start = end
        end += 21
        (_x.x, _x.y, _x.z, _x.doppler, _x.power, _x.track,) = _get_struct_5fB().unpack(str[start:end])
        self.points.append(val1)
      _x = self
      start = end
      end += 8
      (_x.firmware_version, _x.frame_counter,) = _get_struct_Ii().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sI'%length
      start = end
      s = struct.Struct(pattern)
      end += s.size
      self.time_stamp = numpy.frombuffer(str[start:end], dtype=numpy.uint32, count=length)
      start = end
      end += 4
      (self.track_count,) = _get_struct_I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.track_info = []
      for i in range(0, length):
        val1 = retina_view.msg.Track()
        _x = val1
        start = end
        end += 12
        (_x.track_id, _x.track_state, _x.track_count, _x.track_range, _x.track_angle, _x.track_type, _x.track_activity, _x.track_reliability,) = _get_struct_I2BHb3B().unpack(str[start:end])
        self.track_info.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2i = None
def _get_struct_2i():
    global _struct_2i
    if _struct_2i is None:
        _struct_2i = struct.Struct("<2i")
    return _struct_2i
_struct_5fB = None
def _get_struct_5fB():
    global _struct_5fB
    if _struct_5fB is None:
        _struct_5fB = struct.Struct("<5fB")
    return _struct_5fB
_struct_I2BHb3B = None
def _get_struct_I2BHb3B():
    global _struct_I2BHb3B
    if _struct_I2BHb3B is None:
        _struct_I2BHb3B = struct.Struct("<I2BHb3B")
    return _struct_I2BHb3B
_struct_Ii = None
def _get_struct_Ii():
    global _struct_Ii
    if _struct_Ii is None:
        _struct_Ii = struct.Struct("<Ii")
    return _struct_Ii
