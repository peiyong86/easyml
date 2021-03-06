# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modelweights.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='modelweights.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x12modelweights.proto\"\'\n\x07Weights\x12\r\n\x05value\x18\x01 \x03(\x01\x12\r\n\x05shape\x18\x02 \x03(\x05\"\x89\x01\n\x0cModelWeights\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12+\n\x07weights\x18\x02 \x03(\x0b\x32\x1a.ModelWeights.WeightsEntry\x1a\x38\n\x0cWeightsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x17\n\x05value\x18\x02 \x01(\x0b\x32\x08.Weights:\x02\x38\x01\x62\x06proto3')
)




_WEIGHTS = _descriptor.Descriptor(
  name='Weights',
  full_name='Weights',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Weights.value', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='Weights.shape', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=61,
)


_MODELWEIGHTS_WEIGHTSENTRY = _descriptor.Descriptor(
  name='WeightsEntry',
  full_name='ModelWeights.WeightsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='ModelWeights.WeightsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='ModelWeights.WeightsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=145,
  serialized_end=201,
)

_MODELWEIGHTS = _descriptor.Descriptor(
  name='ModelWeights',
  full_name='ModelWeights',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_name', full_name='ModelWeights.model_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weights', full_name='ModelWeights.weights', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_MODELWEIGHTS_WEIGHTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=64,
  serialized_end=201,
)

_MODELWEIGHTS_WEIGHTSENTRY.fields_by_name['value'].message_type = _WEIGHTS
_MODELWEIGHTS_WEIGHTSENTRY.containing_type = _MODELWEIGHTS
_MODELWEIGHTS.fields_by_name['weights'].message_type = _MODELWEIGHTS_WEIGHTSENTRY
DESCRIPTOR.message_types_by_name['Weights'] = _WEIGHTS
DESCRIPTOR.message_types_by_name['ModelWeights'] = _MODELWEIGHTS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Weights = _reflection.GeneratedProtocolMessageType('Weights', (_message.Message,), dict(
  DESCRIPTOR = _WEIGHTS,
  __module__ = 'modelweights_pb2'
  # @@protoc_insertion_point(class_scope:Weights)
  ))
_sym_db.RegisterMessage(Weights)

ModelWeights = _reflection.GeneratedProtocolMessageType('ModelWeights', (_message.Message,), dict(

  WeightsEntry = _reflection.GeneratedProtocolMessageType('WeightsEntry', (_message.Message,), dict(
    DESCRIPTOR = _MODELWEIGHTS_WEIGHTSENTRY,
    __module__ = 'modelweights_pb2'
    # @@protoc_insertion_point(class_scope:ModelWeights.WeightsEntry)
    ))
  ,
  DESCRIPTOR = _MODELWEIGHTS,
  __module__ = 'modelweights_pb2'
  # @@protoc_insertion_point(class_scope:ModelWeights)
  ))
_sym_db.RegisterMessage(ModelWeights)
_sym_db.RegisterMessage(ModelWeights.WeightsEntry)


_MODELWEIGHTS_WEIGHTSENTRY._options = None
# @@protoc_insertion_point(module_scope)
