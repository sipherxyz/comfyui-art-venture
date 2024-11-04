import { app } from '../../../scripts/app.js';
import { ComfyWidgets } from '../../../scripts/widgets.js';

import {
  addKVState,
  chainCallback,
  hideWidgetForGood,
  addWidgetChangeCallback,
} from './utils.js';

function addTextSwitchCaseWidget(nodeType) {
  chainCallback(nodeType.prototype, 'onNodeCreated', function () {
    const dataWidget = this.widgets.find((w) => w.name === 'switch_cases');
    const delimiterWidget = this.widgets.find((w) => w.name === 'delimiter');
    this.widgets = this.widgets.filter((w) => w.name !== 'condition');

    let conditionCombo = null;

    const updateConditionCombo = () => {
      if (!delimiterWidget.value) return;

      const cases = (dataWidget.value ?? '')
        .split('\n')
        .filter((line) => line.includes(delimiterWidget.value))
        .map((line) => line.split(delimiterWidget.value)[0]);

      if (!conditionCombo) {
        conditionCombo = ComfyWidgets['COMBO'](this, 'condition', [
          ['__default__', ...(cases ?? [])],
        ]).widget;
      } else {
        conditionCombo.options.values = ['__default__', ...cases];
      }
    };

    updateConditionCombo();
    dataWidget.inputEl.addEventListener('input', updateConditionCombo);
    addWidgetChangeCallback(delimiterWidget, updateConditionCombo);
  });
}

app.registerExtension({
  name: 'ArtVenture.TextSwitchCase',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData) return;
    if (nodeData.name !== 'TextSwitchCase') {
      return;
    }

    addKVState(nodeType);
    addTextSwitchCaseWidget(nodeType);
  },
});
