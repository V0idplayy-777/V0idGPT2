import { MODEL_DISPLAY, type ModelName } from '../engine/model';

export type Selection = ModelName | 'paintexe';

const PARAMS: Record<string, string> = {
  potato: '~1.5M',
  toaster: '~5M',
  microwave: '~12M',
  blender: '~28M',
  nuclearfridge: '~52M',
  paintexe: 'diffusion',
};

const ORDER: Selection[] = ['potato', 'toaster', 'microwave', 'blender', 'nuclearfridge', 'paintexe'];

export default function ModelTabs({
  selected,
  onSelect,
  disabled,
}: {
  selected: Selection;
  onSelect: (s: Selection) => void;
  disabled: boolean;
}) {
  return (
    <div className="tabs" role="tablist" aria-label="model selection">
      {ORDER.map((name) => {
        const title = name === 'paintexe' ? 'Paint.exe' : MODEL_DISPLAY[name].title;
        const blurb = name === 'paintexe' ? 'Text-to-image diffusion. Tiny but genuine.' : MODEL_DISPLAY[name].blurb;
        return (
          <button
            key={name}
            role="tab"
            aria-selected={selected === name}
            className={selected === name ? 'tab active' : 'tab'}
            disabled={disabled}
            title={blurb}
            onClick={() => onSelect(name)}
          >
            <span className="tab-name">{title}</span>
            <span className="tab-params">{PARAMS[name]}</span>
          </button>
        );
      })}
    </div>
  );
}
