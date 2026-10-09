import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, LoaderCircle, ScanLine } from "lucide-react";
import App from "./App";
import { sourceImages } from "./imageCache";
import {
  api,
  ApiError,
  configureApi,
  errorText,
  selectProject,
  sessionExpiredEvent,
  setCsrfToken,
  usageChangedEvent,
} from "./api";
import type { HostedSession } from "./types";

export function PublicDocument({ page }: { page: "privacy" | "terms" }) {
  return (
    <main className="public-document">
      <a href="/" className="brand">
        <ScanLine size={24} /> Annotation
      </a>
      <h1>{page === "privacy" ? "Privacy policy" : "Terms of use"}</h1>
      {page === "privacy" ? (
        <>
          <p>
            Annotation is operated by Jemailabs. Questions and data requests can
            be sent to{" "}
            <a href="mailto:support@jemailabs.com">support@jemailabs.com</a>.
          </p>
          <h2>Your account and projects</h2>
          <p>
            We use Google sign-in to receive your account identifier, name,
            profile picture, and email address. We do not receive your Google
            password or request access to your Google Drive. A necessary session
            cookie keeps you signed in.
          </p>
          <p>
            Images, filenames, annotation data, and project settings are stored
            to provide the service. Projects are private to their owner. Images
            are stored in private Cloudflare R2 buckets; account and annotation
            databases run on the service infrastructure.
          </p>
          <h2>Image processing</h2>
          <p>
            When you request segmentation, images and prompts are sent over a
            private network to GPU servers operated for this service. Spanish
            prompts may be translated to English to run the model. Your uploads
            and prompts are not used to train models.
          </p>
          <h2>Retention and deletion</h2>
          <p>
            Projects are retained until you delete them. Deleting a project
            removes your access immediately; copies required by backups may
            remain for about seven days, until scheduled cleanup completes.
            Temporary files, generated exports, and completed inference requests
            and results are scheduled for removal after 24 hours. Requests
            waiting in the queue for 24 hours expire without using your daily
            inference quota. Contact us to request deletion of your account.
          </p>
          <h2>Service operation</h2>
          <p>
            We record operational information necessary to secure the service
            and enforce storage and inference limits. Google provides sign-in
            and Cloudflare provides storage and network services, which also
            process connection information to provide their services.
          </p>
        </>
      ) : (
        <>
          <p>
            Annotation provides browser-based image annotation and optional SAM
            3 segmentation. By using the service, you agree to these terms.
            Contact{" "}
            <a href="mailto:support@jemailabs.com">support@jemailabs.com</a> for
            support.
          </p>
          <h2>Your content</h2>
          <p>
            You retain your rights to uploaded content. Upload only images you
            are authorized to process and store. You permit us to store and
            process that content to provide annotation, export, and backup
            functions.
          </p>
          <h2>Fair use</h2>
          <p>
            Each account currently has 1 GB of storage for images, thumbnails,
            and annotation data, plus 300 SAM 3 requests per day, resetting at
            midnight UTC. Requests may wait in a shared queue. Do not bypass
            quotas, interfere with other users, or upload unlawful content.
          </p>
          <h2>Availability and results</h2>
          <p>
            Model output can contain errors and should be reviewed before use.
            GPU availability and service access are not guaranteed. Export your
            annotations regularly. Limits or service features may change; abuse
            may lead to suspension.
          </p>
          <h2>Privacy</h2>
          <p>
            See the <a href="/privacy">privacy policy</a> for account
            information, processing, and retention details.
          </p>
        </>
      )}
      <footer>
        <a href="/">Back to Annotation</a>
        <a href={page === "privacy" ? "/terms" : "/privacy"}>
          {page === "privacy" ? "Terms" : "Privacy"}
        </a>
      </footer>
    </main>
  );
}

export default function RuntimeApp() {
  const [mode, setMode] = useState<"local" | "hosted" | null>(null);
  const [session, setSession] = useState<HostedSession | null>(null);
  const [error, setError] = useState("");
  const [expired, setExpired] = useState(false);
  const [retry, setRetry] = useState(0);
  const sessionRequest = useRef(0);
  const legalPage =
    window.location.pathname === "/privacy"
      ? "privacy"
      : window.location.pathname === "/terms"
        ? "terms"
        : null;
  useEffect(() => {
    if (legalPage) return;
    let active = true;
    setError("");
    void (async () => {
      const response = await fetch("/api/config", {
        credentials: "same-origin",
        cache: "no-store",
      });
      if (!active) return;
      if (response.status === 404) {
        configureApi("local");
        setMode("local");
        return;
      }
      if (!response.ok)
        throw new Error("Could not connect to Annotation. Please try again.");
      const config = await response.json();
      if (!active) return;
      if (config.mode === "local") {
        configureApi("local");
        setMode("local");
        return;
      }
      if (config.mode !== "hosted")
        throw new Error("The server returned an unsupported configuration.");
      configureApi("hosted");
      const current = await api<HostedSession>("/auth/session");
      if (!active) return;
      setCsrfToken(current.csrf_token);
      setSession(current);
      setMode("hosted");
    })().catch((e) => {
      if (active) setError(errorText(e));
    });
    return () => {
      active = false;
    };
  }, [retry, legalPage]);
  async function refreshSession() {
    const requestId = ++sessionRequest.current;
    try {
      const current = await api<HostedSession>("/auth/session");
      if (requestId !== sessionRequest.current) return;
      if (!current.user) {
        sourceImages.clear();
        setExpired(true);
        return;
      }
      if (session?.user && current.user.id !== session.user.id) {
        sourceImages.clear();
        setExpired(true);
        setError(
          `Sign in again as ${session.user.email} to keep working in this window.`,
        );
        return;
      }
      setCsrfToken(current.csrf_token);
      setSession(current);
      setExpired(false);
      setError("");
    } catch (error) {
      // A connectivity failure must not discard or disable manual editing.
      if (
        requestId === sessionRequest.current &&
        error instanceof ApiError &&
        error.status === 401
      ) {
        sourceImages.clear();
        setExpired(true);
      }
    }
  }
  useEffect(() => {
    if (mode !== "hosted" || !session?.user) return;
    const expire = () => {
      sourceImages.clear();
      sessionRequest.current++;
      setExpired(true);
    };
    const refresh = () => {
      void refreshSession();
    };
    window.addEventListener(sessionExpiredEvent, expire);
    window.addEventListener(usageChangedEvent, refresh);
    window.addEventListener("focus", refresh);
    const timer = setInterval(refresh, 60000);
    return () => {
      window.removeEventListener(sessionExpiredEvent, expire);
      window.removeEventListener(usageChangedEvent, refresh);
      window.removeEventListener("focus", refresh);
      clearInterval(timer);
    };
  }, [mode, session?.user?.id]);
  if (legalPage) return <PublicDocument page={legalPage} />;
  if (!mode)
    return (
      <main className="hosted-landing">
        <div className="hosted-landing-card">
          <ScanLine size={36} />
          <h1>Annotation</h1>
          {error ? (
            <>
              <p className="form-error" role="alert">
                {error}
              </p>
              <button
                className="button primary"
                onClick={() => setRetry((value) => value + 1)}
              >
                Try again
              </button>
            </>
          ) : (
            <p role="status">
              <LoaderCircle className="spin" size={20} /> Connecting…
            </p>
          )}
        </div>
      </main>
    );
  if (mode === "local") return <App />;
  if (!session?.user)
    return (
      <main className="hosted-landing">
        <div className="hosted-landing-card">
          <span className="brand-icon">
            <ScanLine size={35} strokeWidth={1.3} />
          </span>
          <p className="hosted-eyebrow">JEMAILABS · ANNOTATION</p>
          <h1>
            Your images.
            <br />
            Ready to annotate.
          </h1>
          <p>
            Label images with points, boxes, polygons, or a description. SAM 3
            runs for you, directly from your browser.
          </p>
          {new URLSearchParams(window.location.search).has("error") && (
            <p className="form-error" role="alert">
              Google sign-in was not completed. You can try again.
            </p>
          )}
          <a
            className="button primary welcome-cta"
            href="/api/v1/auth/google/login"
          >
            Continue with Google <ArrowUpRight size={18} />
          </a>
          <p className="field-note">
            Private projects · 1 GB storage · 300 SAM 3 requests per day
          </p>
          <footer>
            <a href="/privacy">Privacy policy</a>
            <a href="/terms">Terms of use</a>
            <a href="mailto:jrterven@gmail.com">Contact</a>
          </footer>
        </div>
      </main>
    );
  return (
    <div className="hosted-shell">
      {expired && (
        <div className="session-banner" role="alert">
          <span>
            {error ||
              "Your session needs attention. Your edits are still in this window."}
          </span>
          <a
            href="/api/v1/auth/google/login"
            target="_blank"
            rel="noopener noreferrer"
          >
            Sign in again
          </a>
          <button onClick={() => void refreshSession()}>Refresh session</button>
        </div>
      )}
      <div className="hosted-editor" inert={expired || undefined}>
        <App
          hostedSession={session}
          onLogout={async () => {
            sessionRequest.current++;
            await api("/auth/logout", "POST", {});
            sessionRequest.current++;
            selectProject({});
            setCsrfToken(null);
            setSession({ user: null, usage: null, csrf_token: null });
            setExpired(false);
          }}
        />
      </div>
    </div>
  );
}
