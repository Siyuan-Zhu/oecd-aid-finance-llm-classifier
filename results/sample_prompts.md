# Sample prompts

These are 3 prompts built from the committed sample by `src/prompting.py`, with the k=5 most similar training projects (TF-IDF retrieval, same 3-digit sector) as input/output examples. They show what FLAN-UL2 would receive. Nothing was sent to a model to produce this file.

The prompt is a reconstruction from the 2023 write-up, not the original prompt text, which was not kept. Token counts are character-based estimates, not FLAN-UL2 tokenizer counts; the budget is 2048 input tokens. Examples run from least to most similar, so the closest one sits next to the project being coded.

## Test record 2007004448-1: sector 121 Health, General

Correct answer (not in the prompt): `12182` Medical research. Estimated size: 921 tokens.

```text
This is a reading comprehension and text categorisation task. Read the title and description of an aid project and decide which OECD Creditor Reporting System purpose code describes it.
The project belongs to sector 121 (Health, General). Choose exactly one of these 5-digit purpose codes:

12110 - Health policy and administrative management: Health sector policy, planning and programmes; aid to health ministries, public health administration; institution capacity building and advice; medical insurance programmes; including health system strengthening and health governance; unspecified health activities.
12181 - Medical education/training: Medical education and training for tertiary level services.
12182 - Medical research: General medical research (excluding basic health research and research for prevention and control of NCDs (12382)).
12191 - Medical services: Laboratories, specialised clinics and hospitals (including equipment and supplies); ambulances; dental services; medical rehabilitation. Excludes noncommunicable diseases (123xx).

Answer with the 5-digit code only.

Project: Early Careers R&D Partner Network. To support the engagement in leading health research that benefits individuals disproportionately impacted by diseases, who come from historically economically disadvantaged backgrounds.
Purpose code: 12110

Project: Global Antimicrobial Resistance Innovation Fund (GAMRIF) – Programme costs. The primary purpose of the Global AMR Innovation Fund (GAMRIF) project is for the benefit of low- and middle-income countries (LMICs). Programme costs fund essential activities such as delivery partner events, general management to support better alignment, and technical accuracy across GAMRIF activities.
Purpose code: 12110

Project: Stichting Medical Credit Fund. In 2009, the foundation launched the Medical Credit Fund (MCF) to finance small and medium-sized healthcare facilities in Africa, improving access to quality services for low-income patients. MCF provides flexible local currency debt finance to healthcare providers through risk-sharing partnerships with local financial institutions. Borrowers include hospitals, clinics, diagnostic centres, pharmacies and training institutes. Alongside debt capital, MCF provides healthcare providers with technical assistance to help improve their quality and increase their capacity.
Purpose code: 12110

Project: Leading University for International Cooperation(Sri Lanka). Establishment of the Program in Biomolecular Science and Industrial Biotechnology in the Faculty of Science, University of Colombo and the Innovation Center for Local Biotechnology Industry
Purpose code: 12110

Project: Africa Health Fund (Aureos). Africa Health Fund is targeting small and medium sized companies that help low-income Africans gain access to affordable but high-quality health services. The fund is managed by Aureos Capital.
Purpose code: 12191

Project: APIDC Biotech Fund. Indian SME fund focusing on biotechnology and pharmaseutical industry and similar sectors. 70% of this capital come from local Indian investors. 2/3 of the investments are supposed to be in early stage companies. The fund is managed from Hyderabad, India.
Purpose code:
```

## Test record 2012004303-1: sector 122 Basic Health

Correct answer (not in the prompt): `12230` Basic health infrastructure. Estimated size: 1359 tokens.

```text
This is a reading comprehension and text categorisation task. Read the title and description of an aid project and decide which OECD Creditor Reporting System purpose code describes it.
The project belongs to sector 122 (Basic Health). Choose exactly one of these 5-digit purpose codes:

12220 - Basic health care: Basic and primary health care programmes; paramedical and nursing care programmes; supply of drugs, medicines and vaccines related to basic health care; activities aimed at achieving universal health coverage.
12230 - Basic health infrastructure: District-level hospitals, clinics and dispensaries and related medical equipment; excluding specialised hospitals and clinics (12191).
12240 - Basic nutrition: Micronutrient deficiency identification and supplementation; Infant and young child feeding promotion including exclusive breastfeeding; Non-emergency management of acute malnutrition and other targeted feeding programs (including complementary feeding); Staple food fortification including salt iodization; Nutritional status monitoring and national nutrition surveillance; Research, capacity building, policy development, monitoring and evaluation in support of these interventions. Use code 11250 for school feeding and 43072 for household food security.
12250 - Infectious disease control: Immunisation; prevention and control of infectious and parasite diseases, except malaria (12262), tuberculosis (12263), COVID-19 (12264), HIV/AIDS and other STDs (13040). It includes diarrheal diseases, vector-borne diseases (e.g. river blindness and guinea worm), viral diseases, mycosis, helminthiasis, zoonosis, diseases by other bacteria and viruses, pediculosis, etc.
12261 - Health education: Information, education and training of the population for improving health knowledge and practices; public health and awareness campaigns; promotion of improved personal hygiene practices, including use of sanitation facilities and handwashing with soap.
12262 - Malaria control: Prevention and control of malaria.
12263 - Tuberculosis control: Immunisation, prevention and control of tuberculosis.
12264 - COVID-19 control: All activities related to COVID-19 control e.g. information, education and communication; testing; prevention; immunisation, treatment, care.
12281 - Health personnel development: Training of health staff for basic health care services.

Answer with the 5-digit code only.

Project: Balti Regional Hospital. The aim of the Project loan is to partially finance eligible investments for the construction and equipment of a tertiary care hospital for the North Region of Moldova with, amongst other elements, an ambulatory care centre, cancer centre, operating theatres and inpatient wards, with a capacity of 340 inpatient beds and 78 day-beds, providing better and more accessible public tertiary healthcare to the population and offering improved work, study and research conditions for hospital staff and medical students.Final beneficiaries: the residents of the North Region of the Republic of Moldova, and in particular, hospital staff and medical students.Share
Purpose code: 12230

Project: Health Provider Institutions Phase Two. Contributing to implementation of the country's healthcare reform by providing improved conditions for the provision of tertiary health care services. Specifically, the project encompasses the construction and modernisation of the Clinical Centre 'Mother Teresa' and the Regional Clinical Hospital in Shtip.
Purpose code: 12230

Project: Cure Tuberculosis Project 2. The USAID Cure Tuberculosis project assists the Kyrgyz government to diagnose, treat, and cure people with drug-resistant tuberculosis. The project works to strengthen systems, tools, resources, and human capacity to improve the Kyrgyz Republics response to tuberculosis in line with international recommendations.
Purpose code: 12263

Project: Strategic Purchasing for PHC Phase II. To provide tailored support to governments and national stakeholders on foundational primary health care reforms in 6 countries, to ultimately increase health access & outcomes for levels of spend, and to contribute to global learning.
Purpose code: 12220

Project: Supp childhood cancer prog in TZ,Inst collaboratio btwn TLM,Crumlin Childrens Hos. The project supports Child cancer (Paediatric Oncology) services in Tanzania by providing access to quality, free cancer treatment and care services for all children in need.
Purpose code: 12281

Project: Childrens Hospital Establishment Project. 1) To function as a tertiary hospital to cure the diseases of children and increase the health care coverage, 2) to perform an education function for pediatrics interns and conduct research for young childrens diseases
Purpose code:
```

## Test record 2018011648-1: sector 123 Non-communicable diseases (NCDs)

Correct answer (not in the prompt): `12340` Promotion of mental health and well-being. Estimated size: 1576 tokens.

```text
This is a reading comprehension and text categorisation task. Read the title and description of an aid project and decide which OECD Creditor Reporting System purpose code describes it.
The project belongs to sector 123 (Non-communicable diseases (NCDs)). Choose exactly one of these 5-digit purpose codes:

12310 - NCDs control, general: Programmes for the prevention and control of NCDs which cannot be broken down into the codes below.
12340 - Promotion of mental health and well-being: Promotion of programmes and interventions which support mental health and well-being resiliency; prevention, care and support to individuals vulnerable to suicide. Excluding treatment of addiction to tobacco, alcohol and drugs (included in codes 12320 and 12330).
12350 - Other prevention and treatment of NCDs: Population/individual measures to reduce exposure to unhealthy diets and physical inactivity and to strengthen capacity for prevention, early detection, treatment and sustained management of NCDs including: Cardiovascular disease control: Prevention, screening and treatment of cardiovascular diseases (including hypertension, hyperlipidaemia, ischaemic heart diseases, stroke, rheumatic heart disease, congenital heart disease, heart failure, etc.). Diabetes control: Prevention, screening, diagnosis, treatment and management of complications from all types of diabetes. Exposure to physical inactivity: Promotion of physical activity through supportive built environment (urban design, transport), sports, health care, schools and community programmes and mass media campaign. Exposure to unhealthy diet: Programmes and interventions that promote healthy diet through reduced consumption of salt, sugar and fats and increased consumption of fruits and vegetables e.g. food reformulation, nutrient labelling, food taxes, marketing restriction on unhealthy foods, nutrition education and counselling, and settings-based interventions (schools, workplaces, villages, communities). Cancer control: Prevention (including immunisation, HPV and HBV), early diagnosis (including pathology), screening, treatment (e.g. radiotherapy, chemotherapy, surgery) and palliative care for all types of cancers. Implementation, maintenance and improvement of cancer registries are also included. Chronic respiratory diseases: Prevention, early diagnosis and treatment of chronic respiratory diseases, including asthma. Excludes: Tobacco use control (12320), Control of harmful use of alcohol and drugs (12330), research for the prevention and control of NCDs (12382).
12382 - Research for prevention and control of NCDs: Research to enhance understanding of NCDs, their risk factors, epidemiology, social determinants and economic impact; translational and implementation research to enhance operationalisation of cost-effective strategies to prevent and control NCDs; surveillance and monitoring of NCD mortality, morbidity, risk factor exposures, and national capacity to prevent and control NCDs.

Answer with the 5-digit code only.

Project: Reinforcing the access to Mental Health services, key interventions to strengthen individual and community resilience in a protracted crisis. The resilience of the population living under occupation and constant violence in the oPT is strengthened by the improved availability, accessibility, acceptance and quality of the mental health services (MH).
Purpose code: 12340

Project: Mental Health Work Force Training Program. To Promote mental health, prevent the occurrence of mental health problems in vulnerable groups. To exchange knowledge, share experiences and learnings on mental health promotion, prevention, rehabilitation, and treatment in their country. To understand and learn mental health processes, strategies, operations, community mental health services.
Purpose code: 12340

Project: Youth Mental Health Well-being - The Path to a Sustainable Future. The project aims to improve the lives of young people at risk for mental health and mental illness in the Elbasan region by offering training and best practices for caring for the mental health of young people. Duration - 1 year, Budget - BGN 24 305.
Purpose code: 12340

Project: SAKI-M-H-U (Schooling, education, children, Mental health, Ukraine). Training project to train pediatricians, child psychiatrists, child psychologists and child health nurses in the diagnosis and treatment of war-related mental disorders in Ukrainian children/adolescents
Purpose code: 12340

Project: Mental Health for All. In Ukraine, socio-economic factors, the protracted conflict, coupled with an oudated mental health care system result in a high prevalence and poor treatment of mental health disorders. This project supports the implementation of reforms in the mental healthcare sector by improving framework conditions, piloting integrated, communiry-based models of care and providing psychosocial support to communities affected by the ongoing conflict.
Purpose code: 12340

Project: Mental Health for Ukraine (MH4U). The Mental Health for Ukraine project is a flagship Swiss engagement which strengthens the capacity and resilience of the mental health system in a context of growing demand and resource scarcity. Switzerland renews its multi-year commitment and pursues a tailored approach addressing acute mental health needs as a consequence of the conflict. The project contributes to the overall resilience of the Ukrainian people, aligns with government priorities and supports Ukraine's reform path.
Purpose code:
```
