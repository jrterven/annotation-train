import { useState } from "react";
import { UserRound } from "lucide-react";
import type { HostedSession } from "./types";

export default function AccountAvatar({
  user,
}: {
  user: HostedSession["user"];
}) {
  const [failed, setFailed] = useState<string>();
  const picture = user?.picture;
  return (
    <span className="account-avatar" aria-hidden="true">
      {picture && failed !== picture ? (
        <img
          src={picture}
          alt=""
          referrerPolicy="no-referrer"
          onError={() => setFailed(picture)}
        />
      ) : (
        <UserRound size={20} />
      )}
    </span>
  );
}
