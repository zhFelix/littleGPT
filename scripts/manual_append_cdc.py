#!/usr/bin/env python3
"""手动抓取 CDC 两条(requests 被 403 拦截)，用浏览器取得正文后，
复用 fetch_superconductivity.py 的同款 split_sentences / chunk_sentences 切块，追加到 train/article.jsonl。"""
import json
import re

OUT_FILE = "train/article.jsonl"

TARGET = 230
MIN_LEN = 140
MAX_LEN = 380
MIN_SENT = 25

# 源元数据 + 已清洗的正文（从浏览器渲染后的 <main> 内文本手工去除导航/页脚噪音）
SOURCES = [
    {
        "id": "science-en-007",
        "title": "Explaining How Vaccines Work",
        "url": "https://www.cdc.gov/vaccines/basics/explaining-how-vaccines-work.html",
        "source": "CDC",
        "domain": "biology",
        "topic": "how_vaccines_work",
        "text": """Vaccines help the body learn how to defend itself from disease without the dangers of a full-blown infection. The immune response to a vaccine might cause tiredness and discomfort for a day or two, but the resulting protection can last a lifetime.

Infections are unpredictable and can have long-term consequences. Even mild or symptom-less infections can be deadly. For example, most people infected with the human papillomavirus (HPV) never show any sign of infection. But for some, the sign appears years later as an aggressive, life-threatening cancer. By then, it's too late to get vaccinated.

Vaccines work by imitating an infection—the presence of a disease-causing organism in the body—to engage the body's natural defenses. The active ingredient in all vaccines is an antigen, the name for any substance that causes the immune system to begin producing antibodies. In a vaccine, the antigen could be either weakened or killed bacteria or viruses, bits of their exterior surface or genetic material, or bacterial toxin treated to make it non-toxic.

Antibodies are proteins produced by white blood cells to identify and neutralize foreign substances. White blood cells are created in the bone marrow but dispersed throughout the body in low numbers, ready to begin multiplying and attacking microbes and substances not native to the body. After they have eliminated an infection, white blood cells stop multiplying and their numbers dwindle until only a few are left to keep watch. At that point, a person is considered immunized.

Because immunity can take weeks to develop after vaccination, it is possible to become infected in the weeks immediately following vaccination. Even after that, vaccinated people can and sometimes do get infected. But a vaccinated person is far less likely to die or become seriously ill than someone whose immune system is unprepared to fight an infection.

A single dose of vaccine provides only partial protection. The number of doses needed to achieve immunity depends on whether the antigen in a vaccine is alive or not. Because they contain living bacteria or viruses, live-attenuated vaccines can provide enduring protection with only two doses. By contrast, non-live vaccines typically require at least three doses to achieve protection that fades over time and must be restored with booster doses.

Live-attenuated vaccines offer long-lasting, even lifetime protection. They could cause a life-threatening infection in someone with a weak or suppressed immune system, and they require two doses to achieve maximum immunity. Examples include the chickenpox vaccine and the MMR (measles, mumps, and rubella) combined vaccine, which children should receive around their first and fifth birthdays.

Non-live vaccines provide protection that fades over time and are safer for people with weak immune systems. They require three or more doses to achieve maximum immunity. For example, the DTaP vaccine requires repeated doses to achieve and maintain protection from diphtheria, tetanus, and pertussis (whooping cough): infants receive doses at 2 months, 4 months, 6 months, and 18 months of age; children get one booster dose around the time they first enter school and another when they begin middle school; and adults should get a tetanus booster once every 10 years or during each pregnancy.

Certain vaccines must be updated periodically to protect against mutation-prone viruses that cause waves of infections months or years apart. To stay protected, people must get the updated vaccines even if they got an earlier version. The seasonal flu vaccine is reformulated each year to target the four strains expected to be most common and most dangerous. The updated COVID-19 vaccines were developed to deal both with fading immunity and a fast-evolving virus.

History shows that vaccines are the safest, most effective way to protect yourself and your family from many preventable diseases. Everyone should get all recommended vaccines at the recommended times. It is especially important for children and adolescents to get catch-up doses of any missed vaccines or vaccine doses as soon as they can. Adults should get all recommended vaccines for their age or other risk factors such as health condition or occupation.

To be immune is to be partially or fully resistant to a specific infectious disease or disease-causing organism. A person who is immune can resist the bacteria or viruses that cause a disease, but the protection is never perfect. Immunization is the process of being made resistant to an infectious disease, usually by means of a vaccine. Immunity is protection against a disease, and it can be passive or active, natural or vaccine induced.

Active immunity comes from being exposed to a disease-causing organism. Natural immunity results from being infected by a disease-causing organism, whether the infection is symptomatic or not. Vaccine-induced immunity results from being exposed to killed or weakened bacteria or viruses, or even just important pieces of them, through vaccination. Either way, active immunity takes longer to develop but lasts longer than passive immunity.

Passive immunity is provided by antibodies produced by another human being or animal. Full-term babies acquire passive immunity from their mother's antibodies during the final months of pregnancy. Patients can acquire passive immunity through antibody-containing blood products derived from human or animal sources. Passive immunity provides protection that is immediate but fades within weeks or months.""",
    },
    {
        "id": "science-en-008",
        "title": "About Antimicrobial Resistance",
        "url": "https://www.cdc.gov/antimicrobial-resistance/about/index.html",
        "source": "CDC",
        "domain": "biology",
        "topic": "antimicrobial_resistance",
        "text": """Antimicrobial resistance (AR) happens when germs like bacteria and fungi develop the ability to defeat the drugs designed to kill them. That means the germs are not killed and continue to grow. Resistant infections can be difficult, and sometimes impossible, to treat.

Antimicrobial resistance is an urgent global public health threat, killing at least 1.27 million people worldwide and associated with nearly 5 million deaths in 2019. In the United States, more than 2.8 million antimicrobial-resistant infections occur each year, and more than 35,000 people die as a result, according to CDC's 2019 Antibiotic Resistance Threats Report. When Clostridioides difficile, a germ that is not typically resistant to antibiotics but can cause deadly diarrhea and inflammation of the colon and is associated with antimicrobial use, is added to these, the U.S. toll of all the threats in the report exceeds 3 million infections and 48,000 deaths.

Bacteria and fungi do not have to be resistant to every antibiotic or antifungal to be dangerous. Resistance to even one antibiotic or antifungal drug can mean serious problems. For example, antimicrobial-resistant infections that require the use of second- and third-line treatments can harm patients by causing serious side effects, such as organ failure, and prolong care and recovery, sometimes for months. Many medical advances are dependent on the ability to fight infections using antibiotics, including joint replacements, organ transplants, cancer therapy, and the treatment of chronic diseases like diabetes, asthma, and rheumatoid arthritis. In some cases, antimicrobial-resistant infections have no treatment options. If antibiotics and antifungals lose their effectiveness, then we lose the ability to treat infections and control these public health threats.

Germs are microbes, very small living organisms including bacteria, fungi, parasites and viruses. Pathogens are germs that cause infections, and most germs are harmless and even helpful to people. Bacteria cause infections such as strep throat, foodborne illnesses, and other serious infections. Fungi cause infections like athlete's foot, yeast infections, and other serious infections. Antibiotics are medicines that fight infections caused by bacteria in humans and animals by either killing the bacteria or making it difficult for the bacteria to grow and multiply. Antifungals are medicines that treat fungal infections by killing or stopping the growth of dangerous fungi in humans, animals or plants.

Antimicrobial resistance is a naturally occurring process. However, increases in antimicrobial resistance are driven by a combination of germs exposed to antibiotics and antifungals, and the spread of those germs and their resistance mechanisms. Antibiotics and antifungals kill some germs that cause infections, but they also kill helpful germs that protect our body from infection. Antimicrobial resistance accelerates when antibiotics and antifungals pressure bacteria and fungi to adapt. The antimicrobial-resistant germs survive, multiply and spread to other germs. These surviving germs have resistance traits in their DNA that can spread to other germs.

To survive, germs can develop defense strategies against antibiotics and antifungals called resistance mechanisms. DNA tells the germ how to make specific proteins, which determine the germ's resistance mechanisms. Bacteria and fungi can carry genes for many types of resistance. Alarmingly, antimicrobial-resistant germs can share their resistance mechanisms with other germs that have not been exposed to antibiotics or antifungals.

Some germs restrict access of the antibiotic by changing the entryways or limiting the number of entryways. For example, Gram-negative bacteria have an outer layer (membrane) that protects them from their environment, and they can use this membrane to selectively keep antibiotic drugs from entering. Some germs get rid of antibiotics using pumps in their cell walls to remove antibiotic drugs that enter the cell. For example, some Pseudomonas aeruginosa bacteria can produce pumps to get rid of several different important antibiotic drugs, including fluoroquinolones, beta-lactams, chloramphenicol, and trimethoprim. Some Candida species produce pumps that get rid of azoles such as fluconazole.

Some germs change or destroy antibiotics with enzymes, proteins that break down the drug. For example, Klebsiella pneumoniae bacteria produce enzymes called carbapenemases, which break down carbapenem drugs and most other beta-lactam drugs. Many antibiotic drugs are designed to single out and destroy specific parts (or targets) of a bacterium. Germs change the antibiotic's target so the drug can no longer fit and do its job. For example, Escherichia coli bacteria with the mcr-1 gene can add a compound to the outside of the cell wall so that the drug colistin cannot latch onto it. Aspergillus fumigatus changes the cyp1A gene so that triazoles cannot bind to the protein. Some germs develop new cell processes that avoid using the antibiotic's target. For example, some Staphylococcus aureus bacteria can bypass the drug effects of trimethoprim.

Antimicrobial resistance occurs when germs defeat the antibiotic or antifungal drugs designed to kill them. It does NOT mean your body is resistant to antibiotics or antifungals. Antimicrobial resistance can affect people at any stage of life. Infections caused by resistant germs are difficult, sometimes impossible, to treat, and in many cases these infections require extended hospital stays, additional follow-up doctor visits, and treatments that may be costly and potentially toxic.

Healthy habits can protect you from infections and help stop germs from spreading. You should talk to your healthcare provider or veterinarian about whether antibiotics or antifungals are needed, since antibiotics and antifungals do not work on viruses, such as colds and the flu. These drugs save lives but can lead to side effects and antimicrobial resistance. If you have been taking these drugs, tell your doctor if you have three or more diarrhea episodes in 24 hours. You should also tell your healthcare provider if you recently traveled to or received care in another country, because antimicrobial resistance has been found in all regions of the world, and modern trade and travel mean it can move easily across borders and can spread in places like hospitals, farms, the community and the environment.

Antimicrobial resistance is one of the world's most urgent public health problems because it can affect people at any stage of life, affects healthcare, veterinary, and agriculture industries, and can make all antibiotics or antifungals ineffective, resulting in unstoppable infections. Stopping the spread of antimicrobial resistance is crucial. We all have a role to play, from travelers, animal owners and caregivers to patients and healthcare providers, by preventing infections in the first place, improving antibiotic and antifungal use to slow the development of resistance, and stopping the spread of antimicrobial resistance when it does develop.""",
    },
]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n\n", " ").replace("\n", " "))
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < MIN_SENT:
            continue
        if re.search(r"(^related links|^tags:|^subscribe|cookie|all rights reserved|share this)", p, re.I):
            continue
        out.append(p)
    return out


def chunk_sentences(sents: list[str]) -> list[str]:
    chunks = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= MAX_LEN and (len(cur) >= MIN_LEN or len(cur) + 1 + len(s) <= TARGET + 60):
            cur = cur + " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur.strip():
        chunks.append(cur)
    return chunks


def main() -> None:
    records = []
    for src in SOURCES:
        chunks = chunk_sentences(split_sentences(src["text"]))
        for idx, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "id": f"{src['id']}-{idx:02d}",
                    "split": "train",
                    "language": "en",
                    "domain": src["domain"],
                    "topic": src["topic"],
                    "text": chunk,
                    "source": src["source"],
                    "source_url": src["url"],
                    "source_title": src["title"],
                    "chunk_index": idx,
                }
            )
        print(f"{src['id']}: {len(chunks)} chunks")

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"共追加 {len(records)} 条 -> {OUT_FILE}")
    # 汇总每个源的块长
    for src in SOURCES:
        cls = [r for r in records if r["id"].startswith(src["id"])]
        lens = [len(r["text"]) for r in cls]
        print(f"  {src['id']}: {len(cls)} 块, 长度 min/中位/max = {min(lens) if lens else '-'}/{sorted(lens)[len(lens)//2] if lens else '-'}/{max(lens) if lens else '-'}")


if __name__ == "__main__":
    main()