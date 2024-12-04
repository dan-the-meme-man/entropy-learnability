import os
import json
import plotly.express as px

jsons = [
    os.path.join('results', x) for x in os.listdir('results') if x.endswith('.json')
]

for json_file in jsons:
    with open(json_file, 'r') as f:
        data = json.load(f)
        try:
            nested_losses = data['train_losses']
            flat_losses = []
            for losses in nested_losses:
                flat_losses.extend(losses)
            setting = os.path.splitext(os.path.basename(json_file))[0]
            fig = px.scatter(
                x=range(len(flat_losses)),
                y=flat_losses,
                title=f'Loss for {setting}',
                labels={'x': 'Training step', 'y': 'Loss'},
            )
            fig.write_html(os.path.join('plots', f'{setting}.html'))
        except KeyError:
            print(f'Error in {json_file}')