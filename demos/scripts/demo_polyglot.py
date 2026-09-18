"""openjev-lm multilingual demo — postings in languages the model never saw.

Training corpus is English-only. Each row below is a real-language posting
(FR/DE/ES/PT/IT/NL/PL/JA) across the four buckets; the 0.5B adapter scores
the same seven questions anyway. Truth labels shown for honesty.
"""
import sys, time
sys.path.insert(0, '/home/ubuntu/repos/openjev-lm/openjev')
from dataset import prompt_for
import eval_likelihood as E

POSTS = [
    ("FR", "Développeur Full-Stack CDI", "TechCorp Lyon", "55 000 €/an", "staff_role/generic_job"),
    ("FR", "Besoin d'un site vitrine pour mon restaurant", "Chez Marcel", "800 €", "service_lead"),
    ("DE", "Wir suchen Pflegekräfte (m/w/d) Nachtdienst", "Klinikum Nord", "3.400 €/Monat", "generic_job"),
    ("DE", "Suche jemanden für meine Shopify-Seite", "Kleines Café", "1.200 €", "service_lead"),
    ("ES", "Se busca cocinero tiempo completo", "Restaurante El Faro", "1.500 €/mes", "generic_job"),
    ("ES", "Necesito una web para mi taller mecánico", "Talleres Rápidos", "$900", "service_lead"),
    ("PT", "Preciso de alguém para fazer meu site", "Padaria Central", "R$2.000", "service_lead"),
    ("PT", "Enfermeiro(a) plantão noturno — contratação", "Hospital Sul", "R$8.500/mês", "staff_role/generic_job"),
    ("IT", "Cercasi cuoco esperto per ristorante", "Trattoria Roma", "1.600 €/mese", "generic_job"),
    ("NL", "Gezocht: iemand die onze webshop kan bouwen", "Fietsenmaker Jan", "€1.000", "service_lead"),
    ("PL", "Szukam kogoś do zrobienia strony firmowej", "Warsztat Auto", "2.500 zł", "service_lead"),
    ("JA", "Webサイトをってくれるをしています", "さなパン", "¥150,000", "service_lead"),
]

def main():
    print('training corpus: 100% English. these postings: zero English.\n')
    model, tok, _ = E.load_model('/home/ubuntu/repos/openjev-lm/models/Qwen2.5-0.5B-Instruct', 8, want_lora=True)
    state, step = E.load_lora_state('/home/ubuntu/repos/openjev-lm/openjev/runs/retrain-check/adapter.pt')
    print(E.apply_adapter(model, state), f'(adapter step {step})\n')
    stats = {"boundary_fallbacks": 0}
    hits = 0
    for i, (lang, title, emp, pay, want) in enumerate(POSTS):
        row = {"id": f'{lang}{i}', "title": title, "prompt": prompt_for(title, emp, pay)}
        t = time.time()
        out = E.score_row(model, tok, row, stats, use_enum=False)
        s = out['staged']
        sec = time.time() - t
        ok = s['bucket'] in want.split('/')
        hits += ok
        print(f'[{lang}] {title[:46]:<46} → {s["bucket"]:<13} fit={s["fit"]} ({sec:3.1f}s) '
              f'{"✓" if ok else "✗ want " + want}')
        time.sleep(1.0)
    print(f'\n{hits}/{len(POSTS)} correct on never-seen languages — zero fine-tuning, $0')

if __name__ == '__main__':
    main()
