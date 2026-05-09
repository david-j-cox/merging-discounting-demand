/**
 * Consent landing page (Phase 4 step 1).
 *
 * Structure follows the IRB-approved DEGap consent template (Endicott
 * College, 2025): Introduction, Participation, Risks, Benefits,
 * Anonymity, Contact, Electronic Consent. The placeholder fields
 * (institution, researcher contact, IRB contact) **must be replaced**
 * with values from your actual IRB approval before deploying — see
 * the constants block below.
 *
 * Participants must check the agreement box before they can advance.
 * Refusal closes the window.
 */

import jsPsychHtmlButtonResponse from "@jspsych/plugin-html-button-response";

import { STUDY_NAME } from "../config";

// ---------------------------------------------------------------------------
// Study contact and IRB constants (Endicott College).
// ---------------------------------------------------------------------------

const INSTITUTION_NAME = "Endicott College";

const RESEARCHER_CONTACT = {
  name: "David J. Cox, PhD, MSB, BCBA-D",
  email: "dcox@endicott.edu",
};

const IRB_CONTACT = {
  board_name: "Institutional Review Board (IRB)",
  institution: "Endicott College",
  address_line_1: "376 Hale Street",
  address_line_2: "Beverly, MA 01915",
  email: "irb@endicott.edu",
};

/** Compensation summary — Prolific pays out at the posted rate. */
const COMPENSATION_TEXT =
  "Participants recruited via Prolific will be compensated at approximately " +
  "$12 per hour, prorated for actual session length.";

// ---------------------------------------------------------------------------

const CONSENT_HTML = `
  <h1 style="text-align: center;">Informed Consent</h1>

  <div style="background: #f7f7f7; padding: 1.5rem; border-radius: 8px;
              max-width: 720px; margin: 0 auto; text-align: left;
              font-size: 0.95rem; line-height: 1.5;">

    <h2>Introduction</h2>
    <p>
      You are invited to participate in a research study examining how
      people make choices that involve effort and reward. The purpose of
      this research is to understand how the perceived value of a reward
      changes with the amount of effort needed to obtain it.
    </p>
    <p>
      In this study you will (a) calibrate your maximum sustained
      key-press rate, (b) answer questions about how many
      <em>snack credits</em> you would buy at a series of prices,
      (c) choose between an immediate small reward and a larger reward
      that would require a sustained key-press effort, and (d) answer
      questions about how many snack credits you would acquire at a
      series of effort costs.
    </p>
    <p>
      Throughout this study, a <strong>snack credit</strong> is
      hypothetically redeemable for one small packaged snack of your
      choice (for example, a candy bar, a small bag of chips, or
      similar). All choices in this study are hypothetical; no actual
      snacks will be exchanged.
    </p>

    <h2>Participation</h2>
    <p>
      Taking part in this study is completely voluntary. You may stop
      your participation at any time without penalty by closing the
      browser window. There are no right or wrong answers. The full
      session takes about 20 minutes.
    </p>

    <h2>Risks</h2>
    <p>
      There are no foreseeable risks beyond those of normal day-to-day
      computer use. A small fraction of trials require ~30 seconds of
      sustained key-pressing; if at any point you feel discomfort, you
      may stop the trial and the rest of the study without penalty.
    </p>

    <h2>Benefits</h2>
    <p>
      There are no direct benefits to you. The benefits of this
      research are to advance scientific understanding of how people
      trade off effort against reward.
    </p>

    <h2>Compensation</h2>
    <p>${COMPENSATION_TEXT}</p>

    <h2>Anonymity</h2>
    <p>
      Your responses will be stored using anonymous identifiers. No
      personally identifying information will be linked to your
      behavioural data. If you arrived through Prolific, your Prolific
      ID is recorded for payment processing only and is stored
      separately from your responses. Your name or identity will not
      be used in any reports or presentations of this research.
    </p>
    <p>
      This research project has been reviewed by the
      ${IRB_CONTACT.board_name} at ${INSTITUTION_NAME}
      in accordance with US Department of Health and Human Services
      Office of Human Research Protections 45 CFR part 46.
    </p>

    <h2>Contact</h2>
    <p>
      If you have questions about this research, please contact
      ${RESEARCHER_CONTACT.name} at
      <a href="mailto:${RESEARCHER_CONTACT.email}">${RESEARCHER_CONTACT.email}</a>.
    </p>
    <p>
      For questions about your rights as a research participant,
      please contact:
      <br />${IRB_CONTACT.board_name}
      <br />${IRB_CONTACT.institution}
      <br />${IRB_CONTACT.address_line_1}
      <br />${IRB_CONTACT.address_line_2}
      <br /><a href="mailto:${IRB_CONTACT.email}">${IRB_CONTACT.email}</a>
    </p>

    <h2>Electronic Consent</h2>
    <p>
      Clicking <strong>I agree</strong> below confirms that you are
      at least 18 years old, that you have read and understood the
      information above, and that you voluntarily agree to participate
      in this study.
    </p>
  </div>
`;

/**
 * Single jsPsych trial that displays the consent text with two buttons.
 *
 * The trial captures whether the participant agreed; if they decline,
 * the orchestrator (main.ts) shows a polite exit message and stops.
 */
export const CONSENT_TRIAL = {
  type: jsPsychHtmlButtonResponse,
  stimulus: CONSENT_HTML,
  choices: ["I agree", "I do not agree"],
  // No keyboard shortcut — explicit click required.
  on_finish: (data: Record<string, unknown>) => {
    data.consent = {
      agreed: data.response === 0,
      study: STUDY_NAME,
      timestampIso: new Date().toISOString(),
    };
  },
};

/** Did the participant agree? Read after the consent trial finishes. */
export function readConsentAgreement(jsPsychData: Array<Record<string, unknown>>): boolean {
  for (const row of jsPsychData) {
    const consent = row.consent as { agreed?: boolean } | undefined;
    if (consent && typeof consent.agreed === "boolean") return consent.agreed;
  }
  return false;
}
