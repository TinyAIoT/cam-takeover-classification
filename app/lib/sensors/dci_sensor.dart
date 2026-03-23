import 'dart:math';
import 'package:sensebox_bike/blocs/ble_bloc.dart';
import 'package:sensebox_bike/blocs/geolocation_bloc.dart';
import 'package:sensebox_bike/blocs/recording_bloc.dart';
import 'package:sensebox_bike/sensors/sensor.dart';
import 'package:sensebox_bike/services/isar_service.dart';
import 'package:flutter/material.dart';
import 'package:sensebox_bike/ui/widgets/sensor/sensor_display_card.dart';
import 'package:sensebox_bike/ui/widgets/sensor/sensor_value_display.dart';
import 'package:sensebox_bike/ui/widgets/common/sensor_conditional_rerender.dart';
import 'package:sensebox_bike/utils/sensor_utils.dart';
import 'package:sensebox_bike/l10n/app_localizations.dart';

class DCISensor extends Sensor {
  List<double> _latestValue = [0.0];

  static int get staticUiPriority => 10;

  @override
  get uiPriority => staticUiPriority;

  static const String sensorCharacteristicUuid =
      'a12fe5d8-c7f6-4a1d-9bc8-3bf47b9ca25e';

  DCISensor(BleBloc bleBloc, GeolocationBloc geolocationBloc,
      RecordingBloc recordingBloc, IsarService isarService)
      : super(sensorCharacteristicUuid, "dci", [], bleBloc, geolocationBloc,
            recordingBloc, isarService);

  @override
  void onDataReceived(List<double> data) {
    super.onDataReceived(data);
    if (data.isNotEmpty) {
      _latestValue = data;
    }
  }

  @override
  Duration get lookbackWindow => const Duration(milliseconds: 2000);

  @override
  List<double> aggregateData(List<List<double>> valueBuffer) {
    List<double> myValues = valueBuffer.map((e) => e[0]).toList();
    List<double> nonZeroValues =
        myValues.where((value) => value != 0.0).toList();
    if (nonZeroValues.isNotEmpty) {
      return [nonZeroValues.reduce(min)];
    }
    return [0.0];
  }

  @override
  Widget buildWidget() {
    return SensorConditionalRerender(
      valueStream: valueStream,
      initialValue: _latestValue,
      latestValue: _latestValue,
      decimalPlaces: 2,
      builder: (context, value) {
        return SensorDisplayCard(
          title: AppLocalizations.of(context)!.sensorDCI,
          icon: getSensorIcon(title),
          color: getSensorColor(title),
          valueStream: valueStream,
          initialValue: _latestValue,
          decimalPlaces: 2,
          valueBuilder: (context, value) => SensorValueDisplay(
            value: value[0].toStringAsFixed(2),
            unit: '',
            isValid: value[0] != 0.0,
          ),
        );
      },
    );
  }
}
