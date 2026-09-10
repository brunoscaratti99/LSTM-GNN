import urllib
import json
import gzip
from pathlib import Path
import sys

def carregar_mapa_rs_ibge():
    url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/43?formato=application/vnd.geo+json"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, application/geo+json",
            "Accept-Encoding": "identity",
        }
    )

    with urllib.request.urlopen(req) as resposta:
        dados = resposta.read()

    # Se vier compactado em gzip, descompacta antes de decodificar
    if dados[:2] == b"\x1f\x8b":
        dados = gzip.decompress(dados)

    return json.loads(dados.decode("utf-8"))


def salvar_frames_animacao_run(run_dir, *, start_date, end_date, lead_day, **kwargs):
    """Cria ``Predict``/``Real`` para um lead day ao longo de um período.

    Esta função curta mantém o ponto de entrada junto do mapa usado no
    seminário. A implementação mantida fica em ``src/Evaluation``.
    """
    raiz = Path(__file__).resolve().parents[1]
    if str(raiz / "src") not in sys.path:
        sys.path.insert(0, str(raiz / "src"))
    from Evaluation.rs_animation_maps import salvar_frames_animacao_rs

    return salvar_frames_animacao_rs(
        run_dir,
        start_date=start_date,
        end_date=end_date,
        lead_day=lead_day,
        **kwargs,
    )


if __name__ == "__main__":
    salvar_frames_animacao_run(
        r"C:\Local Repository\LSTM-GNN\Experiments\run_experiment\comparative_hidden_dim_20260825_192200\run_015__window_size=45__hidden_dim=64",
        start_date="2023-28-08",
        end_date="2023-31-09",
        lead_day=1,
    )
