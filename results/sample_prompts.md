# Sample prompts

These are 3 prompts built from the committed sample by `src/prompting.py`, with the k=5 most similar training projects (TF-IDF retrieval, same 3-digit sector) as input/output examples. They show what FLAN-UL2 would receive. Nothing was sent to a model to produce this file.

The prompt is a reconstruction from the 2023 write-up, not the original prompt text, which was not kept. Token counts are character-based estimates, not FLAN-UL2 tokenizer counts; the budget is 2048 input tokens. Examples run from least to most similar, so the closest one sits next to the project being coded.

## Test record 2007004448-1: sector 121 Health, General

Correct answer (not in the prompt): `12182` Medical research. Estimated size: 812 tokens.

```text
This is a reading comprehension and text categorisation task. Read the title and description of an aid project and decide which OECD Creditor Reporting System purpose code describes it.
The project belongs to sector 121 (Health, General). Choose exactly one of these 5-digit purpose codes:

12110 - Health policy and administrative management: Health sector policy, planning and programmes; aid to health ministries, public health administration; institution capacity building and advice; medical insurance programmes; including health system strengthening and health governance; unspecified health activities.
12181 - Medical education/training: Medical education and training for tertiary level services.
12182 - Medical research: General medical research (excluding basic health research and research for prevention and control of NCDs (12382)).
12191 - Medical services: Laboratories, specialised clinics and hospitals (including equipment and supplies); ambulances; dental services; medical rehabilitation. Excludes noncommunicable diseases (123xx).

Answer with the 5-digit code only.

Project: Türkiye contribution to the Common Provisioning Fund for the EFSD+ 2024. C(2024)9162 of 17.12.2024 on the financing of the individual measure for the provisioning in 2024 of budgetary guarantees under the European Fund for Sustainable Development Plus (EFSD+) for Türkiye
Purpose code: 12181

Project: Early Careers R&D Partner Network. To support the engagement in leading health research that benefits individuals disproportionately impacted by diseases, who come from historically economically disadvantaged backgrounds.
Purpose code: 12110

Project: Development of next generation anti-malarials targeting the essential parasite protein kinase PfCLK3. MRC IIB award to develop early stage drugs that can offer the potential of curing the symptoms of malaria but also importantly to prevent transmission of malaria from one person to another
Purpose code: 12182

Project: Capacity-building for health systems design in India. to support capacity building of a new cadre of Indian researchers/analysts in health systems design and to conduct relevant health systems research to contribute to improving people's health, well-being and financial protection in India
Purpose code: 12110

Project: Africa Health Fund (Aureos). Africa Health Fund is targeting small and medium sized companies that help low-income Africans gain access to affordable but high-quality health services. The fund is managed by Aureos Capital.
Purpose code: 12191

Project: APIDC Biotech Fund. Indian SME fund focusing on biotechnology and pharmaseutical industry and similar sectors. 70% of this capital come from local Indian investors. 2/3 of the investments are supposed to be in early stage companies. The fund is managed from Hyderabad, India.
Purpose code:
```

## Test record 2018000442006-1: sector 122 Basic Health

Correct answer (not in the prompt): `12240` Basic nutrition. Estimated size: 1460 tokens.

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

Project: emergency nutrition network. ENN's goal is improved outcomes for nutritionally at-risk populations. Project funding to the Emergency Nutrition Network (ENN) to support the implementation of this programme supporting evidence, knowledge management and learning on nutrition for crises affected and high burden contexts, for the delivery of consensus and agreement on actions to improve nutrition outcomes, more effective nutrition policy, programming and institutional architecture, priority research/ technical reviews and related policy influencing, and key findings disseminated in an accessible manner.ENN adopts a gender transformative approach to their work and priortisies adolesence girls and maternal and lactating women in addressing nutriton security.
Purpose code: 12240

Project: Advancing Nutrition. Funding to provide technical support, implement innovative nutrition interventions, and generate evidence to enhance nutrition programming toward achieving the overall goal of improving nutrition for a health.
Purpose code: 12240

Project: Afikepo Nutrition Programme in Malawi Multi-Annual programme Estimate (Afikepo MAPE). The Overall objective of Afikepo Nutrition Programme in Malawi is to enhance nutrition security in Malawi. Afikepo MAPE will allow the DNHA to implement Afikepo activities as foreseen by the FA, i.e.2.1.4 (Afikepo components 2 and 3) trainings, governance coordination, capacity building, programme management and monitoring of all other activities implemented through PAGODA (FAO and GIZ) and Advocacy grant funding (NGOs) and institutionalisation of nutrition M&E framework
Purpose code: 12240

Project: Monitoring, Evaluation, and Learning Activity. The purpose of the Monitoring, Evaluation and Learning (MEL) Platform is to provide technical and advisory services to design and carry out various monitoring, evaluation and learning activities for USAID/Mali.
Purpose code: 12250

Project: Strengthening national nutrition information systems. The overall objective of the proposed initiative is increased uptake of nutrition information and improved knowledge for policy and programme development, implementation and monitoring. The specific objective of this project is improved NIS and country capacity in monitoring programmes and national/international nutrition targets.
Purpose code: 12240

Project: Knowledge and research for nutrition. The overall objective is to provide improved knowledge and evidence for better project/programme design & management, monitoring and evaluation and policy development for nutrition, in view of increasing the effectiveness of programmes and investments in that crucial development area.
Purpose code:
```

## Test record 2022001211-1: sector 123 Non-communicable diseases (NCDs)

Correct answer (not in the prompt): `12340` Promotion of mental health and well-being. Estimated size: 1850 tokens.

```text
This is a reading comprehension and text categorisation task. Read the title and description of an aid project and decide which OECD Creditor Reporting System purpose code describes it.
The project belongs to sector 123 (Non-communicable diseases (NCDs)). Choose exactly one of these 5-digit purpose codes:

12310 - NCDs control, general: Programmes for the prevention and control of NCDs which cannot be broken down into the codes below.
12340 - Promotion of mental health and well-being: Promotion of programmes and interventions which support mental health and well-being resiliency; prevention, care and support to individuals vulnerable to suicide. Excluding treatment of addiction to tobacco, alcohol and drugs (included in codes 12320 and 12330).
12350 - Other prevention and treatment of NCDs: Population/individual measures to reduce exposure to unhealthy diets and physical inactivity and to strengthen capacity for prevention, early detection, treatment and sustained management of NCDs including: Cardiovascular disease control: Prevention, screening and treatment of cardiovascular diseases (including hypertension, hyperlipidaemia, ischaemic heart diseases, stroke, rheumatic heart disease, congenital heart disease, heart failure, etc.). Diabetes control: Prevention, screening, diagnosis, treatment and management of complications from all types of diabetes. Exposure to physical inactivity: Promotion of physical activity through supportive built environment (urban design, transport), sports, health care, schools and community programmes and mass media campaign. Exposure to unhealthy diet: Programmes and interventions that promote healthy diet through reduced consumption of salt, sugar and fats and increased consumption of fruits and vegetables e.g. food reformulation, nutrient labelling, food taxes, marketing restriction on unhealthy foods, nutrition education and counselling, and settings-based interventions (schools, workplaces, villages, communities). Cancer control: Prevention (including immunisation, HPV and HBV), early diagnosis (including pathology), screening, treatment (e.g. radiotherapy, chemotherapy, surgery) and palliative care for all types of cancers. Implementation, maintenance and improvement of cancer registries are also included. Chronic respiratory diseases: Prevention, early diagnosis and treatment of chronic respiratory diseases, including asthma. Excludes: Tobacco use control (12320), Control of harmful use of alcohol and drugs (12330), research for the prevention and control of NCDs (12382).
12382 - Research for prevention and control of NCDs: Research to enhance understanding of NCDs, their risk factors, epidemiology, social determinants and economic impact; translational and implementation research to enhance operationalisation of cost-effective strategies to prevent and control NCDs; surveillance and monitoring of NCD mortality, morbidity, risk factor exposures, and national capacity to prevent and control NCDs.

Answer with the 5-digit code only.

Project: Promotion de la santé mentale et prévention du suicide-NP. FAIRMED and Christoffel Blindenmission (CBM) have joined forces in the LNOB Alliance. It works in fragile contexts, prioritizing the needs of the most marginalized and discriminated people especially of women, men, girls and boys with disabilities. It seeks to improve living conditions of individuals and empowering marginalized communities while also strengthening inclusive systems and services. The Alliance contributes to the implementation of the UN Conventions on the Rights of Persons with Disabilities, which Switzerland ratified in 2014.
Purpose code: 12340

Project: Santé mentale pour tous III (OCADES)-BF. FAIRMED and Christoffel Blindenmission (CBM) have joined forces in the LNOB Alliance. It works in fragile contexts, prioritizing the needs of the most marginalized and discriminated people especially of women, men, girls and boys with disabilities. It seeks to improve living conditions of individuals and empowering marginalized communities while also strengthening inclusive systems and services. The Alliance contributes to the implementation of the UN Conventions on the Rights of Persons with Disabilities, which Switzerland ratified in 2014.
Purpose code: 12340

Project: Santé mentale communautaire-BO. FAIRMED and Christoffel Blindenmission (CBM) have joined forces in the LNOB Alliance. It works in fragile contexts, prioritizing the needs of the most marginalized and discriminated people especially of women, men, girls and boys with disabilities. It seeks to improve living conditions of individuals and empowering marginalized communities while also strengthening inclusive systems and services. The Alliance contributes to the implementation of the UN Conventions on the Rights of Persons with Disabilities, which Switzerland ratified in 2014.
Purpose code: 12340

Project: EQuaLISED-ZW. FAIRMED and Christoffel Blindenmission (CBM) have joined forces in the LNOB Alliance. It works in fragile contexts, prioritizing the needs of the most marginalized and discriminated people especially of women, men, girls and boys with disabilities. It seeks to improve living conditions of individuals and empowering marginalized communities while also strengthening inclusive systems and services. The Alliance contributes to the implementation of the UN Conventions on the Rights of Persons with Disabilities, which Switzerland ratified in 2014.
Purpose code: 12340

Project: Sustainable support to people displaced by conflict and natural disasters and their host communities/ Social Cohesion. The project aims to strengthen social cohesion for forcibly displaced populations and host communities in Ethiopia through an area and community based approach focusing on women and people with disability
Purpose code: 12340

Project: Hemayah: Protecting and Empowering Marginalized Communities. Egypt is a destination country for migrants and forcibly displaced persons facing challenging humanitarian conditions and limited prospects for integration. Switzerland will contribute to the Hemayah/Protection project implemented by Terre des Hommes to improve survival, protection, self-reliance, and wellbeing of migrants and forcibly displaced persons from marginalized communities. The project will provide comprehensive services and opportunities for skills development and socioeconomic integration through a sustainable community based approach.
Purpose code:
```
